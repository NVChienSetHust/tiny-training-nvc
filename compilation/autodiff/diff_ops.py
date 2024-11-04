from pprint import pprint
from pydoc import visiblename
import numpy as np

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import ceil
import tvm
from tvm import relay
from tvm import topi
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from tvm.relay.op.reduce import sum as _sum
from tvm.relay.op import nn as _nn
from tvm.relay.op.tensor import (
    shape_of,
)
from tvm.relay.op.transform import (
    broadcast_to_like,
    collapse_sum_like,
    cast_like,
    reshape,
    reshape_like,
    strided_slice,
    take,
    transpose,
    where,
    repeat,
    expand_dims,
    full_like,
    split,
    squeeze,
    strided_set,
    arange,
    scatter_nd,
)

# import graphviz
from .op2grad import register_gradient, GRAD_OP_MAP
from .diff_ops_bakup import *
from .diff_ops_bakup import _get_reduce_axis, _unreduce_expand

# from graph_tools.visualize_call import visualize_call
def check_call_info(call):
    expr = relay.Function(relay.analysis.all_vars(call), call)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    return mod["main"].body.checked_type


@register_gradient("split")
def split_grad(orig, grad):
    # return [
    #     relay.ones(_.checked_type.shape, dtype=_.checked_type.dtype) for _ in orig.args
    # ]
    """Returns [grad, grad]"""
    # TODO: check why collapse_sum is necessary here
    assert len(orig.args) == 1
    assert isinstance(grad, dict), f"{type(grad)}"

    # print([type(_) for _ in orig.args])
    t = orig.args[0]
    attrs = orig.attrs
    dshape = [int(_) for _ in t.checked_type.shape]
    dtype = t.checked_type.dtype
    # print(dshape, type(dshape))
    # print(attrs.indices_or_sections, attrs.axis)
    indices = [int(_) for _ in attrs.indices_or_sections]
    # print(indices)
    # print(type(indices))
    return_grads = []
    start = 0
    for idx, ind in enumerate(indices):
        if idx in grad:
            return_grads.append(grad[idx])
        else:
            dshape = [int(_) for _ in t.checked_type.shape]
            dshape[attrs.axis] = ind - start
            return_grads.append(relay.zeros(dshape, dtype=dtype))
        start = ind
    idx += 1
    if idx in grad:
        return_grads.append(grad[idx])
    else:
        dshape = [int(_) for _ in t.checked_type.shape]
        dshape[attrs.axis] = dshape[attrs.axis] - start
        return_grads.append(relay.zeros(dshape, dtype=dtype))
    # print(grad[0], len(grad[0]))
    # print(type(return_grads[0]), type(return_grads[1]), type(return_grads[2]))
    # exit(0)
    out_grad = concatenate(return_grads, axis=attrs.axis)
    # [(type(_), _.checked_type.shape, shape_of(_)) for _ in orig.args],
    # print(visualize_call(out_grad))
    return [
        out_grad,
    ]


# TODO: dirty fix for mcu setting
@register_gradient("cast", level=30)
def cast_grad(orig, grad):
    x = orig.args[0]
    return [
        grad,
    ]


# TODO: dirty fix for MCU settings.
@register_gradient("mcumean")
def mcumean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    # dtype = data.checked_type.dtype
    dtype = "float32"
    grad, data = [relay.cast(_, dtype) for _ in (grad, data)]

    if axis is None:
        axis = list(range(len(shape)))

    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]
    # print(shape)
    # print(check_call_info(grad))
    # print(axis)
    # rep = [1 for _ in shape]
    return [
        grad
        * relay.const(mult, dtype=dtype)
        * relay.ones_like(data)
        # relay.tile(grad * const(mult, dtype=dtype), reps=1),
        # relay.tile(grad * const(mult, dtype=dtype), reps=1)
    ]
    return [broadcast_to_like(grad * const(mult, dtype=dtype), data)]


@register_gradient("nn.mcutruncate")
def mcutruncate_grad(orig, grad):
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    x = new_inputs[0]
    dtype = "float32"
    # min = orig.attrs.min
    # max = orig.attrs.max
    min = relay.const(orig.attrs.min, dtype=dtype)
    max = relay.const(orig.attrs.max, dtype=dtype)

    mask1 = relay.greater_equal(x, min)
    mask2 = relay.less_equal(x, max)

    # mask = relay.logical_and(mask1, mask2)
    mask = mask1 * mask2
    zeros = relay.zeros_like(grad)
    # mask = relay.cast(mask, "float32")
    return [
        relay.where(mask, grad, zeros),
    ]


@register_gradient("nn.mcuconv2d")
def mcunetconv2d_grad(orig, grad):
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs

    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
    )
    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]
    return [
        backward_data,
        # relay.zeros_like(o_data),
        backward_weight,
        # relay.zeros_like(o_weight),
        backward_bias,
        # relay.zeros_like(o_bias),
        # backward_zero_x,
        relay.zeros_like(o_zx),
        # backward_zero_y,
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]

def calculate_pool_output_shape(input_shape, pool_size, strides, padding):
    n, c, input_h, input_w = input_shape
    
    output_h = math.floor((input_h - pool_size + 2 * padding) / strides) + 1
    output_w = math.floor((input_w - pool_size + 2 * padding) / strides) + 1
    
    return (n, c, output_h, output_w)

def calculate_sum_shape(input_shape, axes):
    """
    Calculate the shape of the output tensor after summing over specified axes, while keeping it a 4D tensor.

    Parameters:
    - input_shape: Tuple of 4 integers representing the shape of the 4D tensor (N, C, H, W).
    - axes: List or tuple of integers representing the axes to sum over (0, 1, 2, 3).

    Returns:
    - Tuple representing the new shape after summation, still as a 4D tensor.
    """
    # Initialize the output shape with the input shape
    output_shape = list(input_shape)

    # Set the size of the summed dimensions to 1
    for axis in axes:
        output_shape[axis] = 1

    # Ensure the output shape remains a 4D tensor
    return tuple(output_shape)

def calculate_grad_x_sum_shape(weight_sum_shape, grad_y_avg_flat_shape):
    # Transpose weight_sum, so it becomes (576, 160) from (160, 576)
    transposed_weight_shape = (weight_sum_shape[1], weight_sum_shape[0])
    
    # Perform dense operation: result shape is (n, 160) where n comes from grad_y_avg_flat
    n = grad_y_avg_flat_shape[0]
    output_shape = (n, transposed_weight_shape[1])
    
    return output_shape

@register_gradient("nn.mcuconv2davg")
def mcunetconv2davg_grad(orig, grad):
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale, o_order = orig.args

    input_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale, order = new_inputs
    
    # Dirty fix the order = 2

    print("CHECK IN DATA TYPE", o_data.checked_type)
    print("CHECK IN WEIGHT TYPE", o_weight.checked_type)
    
    np_order = 2

    # Get the shape from the checked_type

    attrs = orig.attrs
    n, c_in, in_h, in_w = input_shape
    c_out, _, k_h, k_w = weight_shape
    stride_h, stride_w = get_const_tuple(attrs.strides)
    _, _, padding_h, padding_w = get_const_tuple(attrs.padding)
    d_h, d_w = get_const_tuple(attrs.dilation)

    # Calculate output height and width
    out_h = math.floor((in_h + 2 * padding_h - (d_h * (k_h - 1) + 1)) / stride_h + 1)
    out_w = math.floor((in_w + 2 * padding_w - (d_w * (k_w - 1) + 1)) / stride_w + 1)

    # Perform truncation

    p_h = ceil(in_h / np_order)
    p_w = ceil(in_w / np_order)
    
    print("p_h=", p_h)
    print("p_w=", p_w)

    # Sum weight over last two dimensions and clamp to avoid overflow
    weight_sum = relay.sum(weight, axis=[-1, -2])
    weight_sum = relay.nn.mcutruncate(weight_sum)
    weight_sum = relay.cast(weight_sum, "float32")
    # Calculate the output shape after summing over the last two dimensions
    weight_sum_shape = calculate_sum_shape(weight_shape, axes=[-1, -2])

    x_order_h = (np_order * stride_h)
    x_order_w = (np_order * stride_w)

    x_pad_h, x_pad_w = ceil((p_h * x_order_h - in_h) / 2), ceil((p_w * x_order_w - in_w) / 2)

    print("x_order_h=", x_order_h, "x_order_w=", x_order_w, "x_order_h=", x_order_h, "x_order_w", x_order_w, "x_pad_h=", x_pad_h, "x_pad_w", x_pad_w)

    #TODO: check strides bigger than 1

    # Average pooling with custom divisor
    x_avg = relay.nn.avg_pool2d(data, pool_size=(x_order_h, x_order_w), strides=(x_order_h, x_order_w),
                                padding=(x_pad_h, x_pad_w), ceil_mode=False, count_include_pad=False)

    _, _, x_avg_h, x_avg_w = calculate_pool_output_shape(input_shape, x_order_h, x_order_h, x_pad_h)

    # Multiply the result of average pooling by the kernel size to get the sum
    kernel_area = x_order_h * x_order_w
    x_sum = relay.multiply(x_avg, relay.const(kernel_area, dtype="float32"))

    x_sum_shape = calculate_pool_output_shape(input_shape, x_order_h, x_order_h, x_pad_h)
    # Clamp the result and round it
    x_sum = relay.nn.mcutruncate(x_sum, out_dtype="int8")
    x_sum = relay.cast(x_sum, "float32")

    output_shape = n, c_out, out_h, out_w

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    # backward_zero_y = relay.sum(grad, axis=1, exclude=True)

    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)

    # Compute padding for the gradient pooling operation

    gy_h = out_h
    gy_w = out_w

    grad_y_pad_h = ceil((p_h * np_order - gy_h) / 2)
    grad_y_pad_w = ceil((p_w * np_order - gy_w) / 2)


    # Average pooling for the gradient
    grad_y_avg = relay.nn.avg_pool2d(grad, pool_size=(np_order, np_order), strides=(np_order, np_order),
                                     padding=(grad_y_pad_h, grad_y_pad_w), count_include_pad=False)

    _, _, grad_y_avg_h, grad_y_avg_w = calculate_pool_output_shape(output_shape, np_order, np_order, grad_y_pad_h)
    # Perform backward computation depending on grouping

    if attrs.groups != 1:
        # Grouped convolution case
        #TODO: check dim
        print("before weight sum reshape")
        print("--------------------input shape-----------------------------------")
        print(input_shape)
        print("--------------------x_sum_shape shape-----------------------------------")
        print(x_sum_shape)
        print("-------------------weight shape----------------------")
        print(weight_shape)
        print("-------------------weight_sum shape----------------------")
        print(weight_sum_shape)
        print("--------------------stride------------------")
        print(get_const_tuple(attrs.strides))
        print("--------------------padding------------------")
        print(get_const_tuple(attrs.padding))
        print("------------------dilation-----------------")
        print(get_const_tuple(attrs.dilation))
        print("------------------groups-----------------")
        print(attrs.groups)
        print("--------------------output shape-----------------------------------")
        print(n, c_out, out_h, out_w)
        print("----------------------------grad_y_avg shape------------------------------")
        print(calculate_pool_output_shape(output_shape, np_order, np_order, grad_y_pad_h))
        print("grad_y_pad_h=", grad_y_pad_h, "grad_y_pad_w=", grad_y_pad_w)
        
        # Reshape weight_sum for broadcasting
        weight_sum_reshaped = relay.reshape(weight_sum, (1, c_out, 1, 1))

        # Compute grad_x_sum by multiplying grad_y_avg and weight_sum_reshaped
        grad_x_sum = grad_y_avg * weight_sum_reshaped

        # Compute grad_w_sum by multiplying grad_y_avg and x_sum
        grad_w_sum = grad_y_avg * x_sum

        # Sum grad_w_sum along axes [0, 2, 3] (batch, height, width)
        grad_w_sum = relay.sum(grad_w_sum, axis=[0, 2, 3])

        # Reshape grad_w_sum to (c_out, 1, 1, 1)
        grad_w_sum = relay.reshape(grad_w_sum, (c_out, 1, 1, 1))

        # Use relay.tile instead of broadcast_to
        # Calculate the repetition factor for tiling:
        tile_reps_w = (1, 1, k_h, k_w)  # Repeat across height (k_h) and width (k_w)
        grad_w = relay.tile(grad_w_sum, reps=tile_reps_w)

    else:
        # Non-grouped case (use dense or matmul for matrix multiplication)
        print("before grad_y_avg reshape")
        print("n =", n, "c_out =", c_out, "grad_y_avg_h =", grad_y_avg_h, "grad_y_avg_w =", grad_y_avg_w)

        # grad_y_avg_flat = relay.reshape(grad_y_avg, (n * c_out, grad_y_avg_h * grad_y_avg_w))
        # weight_sum_transposed = relay.transpose(weight_sum, axes=[1, 0, 2, 3])
        
        # grad_x_sum = relay.nn.matmul(weight_sum_transposed, grad_y_avg_flat)

        # print("before grad_x_sum reshape")
        # print("n =", n, "c_in =", c_in, "p_h =", p_h, "p_w =", p_w)
        # grad_x_sum = relay.reshape(grad_x_sum, (n, c_in, p_h, p_w))
        
        # Conv implementation

        # infer output_padding
        fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
            (padding_h, padding_w), (1, 1)
        )

        out_trans_h = (out_h - 1) * stride_h - fpad_top - fpad_bottom + 1
        out_trans_w = (out_w - 1) * stride_w - fpad_left - fpad_right + 1
        output_padding = (in_h - out_trans_h, in_w - out_trans_w)

        #TODO: check transpose conv2d of grad_x_sum

        weight_sum = relay.reshape(weight_sum, (weight_sum_shape[0], weight_sum_shape[1], 1, 1))
        grad_x_sum = _nn.conv2d_transpose(
            grad_y_avg,
            weight_sum,
            strides=attrs.strides,
            padding=attrs.padding,
            dilation=attrs.dilation,
            groups=attrs.groups,
            output_padding=output_padding,
            # to fix codegen bug
            # TODO(lyken17): figure out why missing default value leads to error
            kernel_size=(1, 1),
            channels=c_in,
            )

        # #problem here!!!!
        # gy = relay.reshape(relay.transpose(grad_y_avg, axes=[1, 0, 2, 3]), (c_out, n * grad_y_avg_h * grad_y_avg_w))
        # gx = relay.reshape(relay.transpose(x_sum, axes=[0, 2, 3, 1]), (n * x_avg_h * x_avg_w, c_in))
        
        # grad_w_sum = relay.nn.matmul(gy, gx)
        # grad_w = relay.reshape(grad_w_sum, (c_out, c_in, 1, 1))
        # #TODO: fix the kernel_size !=1 case
        # if k_h != 1:
        #     grad_w = relay.broadcast_to(grad_w, (c_out, c_in, k_h, k_w))

        # 1. Tile grad_y_avg
        grad_y_avg = tile(grad_y_avg, [1, c_in // attrs.groups, 1, 1])

        # 2. Reshape grad_y_avg
        grad_y_avg = reshape(grad_y_avg, [-1, 1, grad_y_avg_h, grad_y_avg_w])

        # 3. Reshape x_sum
        x_sum = reshape(x_sum, [1, -1, x_sum_shape[2], x_sum_shape[3]])

        #TODO: check group conv of GF implementation

        # 4. Apply conv2d operation
        grad_w_sum_conv = _nn.conv2d(
            x_sum,
            grad_y_avg,
            strides=(1, 1),
            padding=(0, 0),
            groups=c_in * n
        )

        # 5. Reshape the result to [n, c_in // groups, c_out, height, width]
        grad_w_sum_conv = reshape(
            grad_w_sum_conv,
            [n, c_in // attrs.groups, c_out, weight_sum_shape[2], weight_sum_shape[3]]
        )

        # 6. Sum the batch dimension (dimension 0)
        grad_w_conv = _sum(grad_w_sum_conv, axis=0)

        # 7. Transpose grad_w_conv from [c_in // groups, c_out, h, w] to [c_out, c_in // groups, h, w]
        grad_w_conv = relay.transpose(grad_w_conv, axes=[1, 0, 2, 3])

        # 8. Slice if grad_w_conv has extra dimensions (greater than kernel size k_h, k_w)
        if weight_sum_shape[2] > k_h or weight_sum_shape[3] > k_w:
            grad_w_conv = relay.strided_slice(grad_w_conv, begin=[0, 0, 0, 0], end=[None, None, k_h, k_w])
        
        if k_h != 1:
            grad_w_conv = relay.broadcast_to(grad_w_conv, (c_out, c_in, k_h, k_w))

        grad_w = grad_w_conv

    #TODO: implement these in C code

    # Reshape and permute gradient for data
    # Step 1: Reshape grad_x_sum to (n, c_in, p_h, p_w, 1, 1)
    grad_x = relay.reshape(grad_x_sum, (n, c_in, p_h, p_w, 1, 1))

    # Step 2: Tile grad_x to (n, c_in, p_h, p_w, order * s_h, order * s_w)
    # Calculate the repetition factor for the tile operation:
    tile_reps = (1, 1, 1, 1, np_order * x_order_h, np_order * x_order_w)
    grad_x_tiled = relay.tile(grad_x, reps=tile_reps)

    # Step 3: Permute axes (0, 1, 2, 4, 3, 5) -> equivalent to relay.transpose
    grad_x_permuted = relay.transpose(grad_x_tiled, axes=(0, 1, 2, 4, 3, 5))

    # Step 4: Reshape to (n, c_in, p_h * order * s_h, p_w * order * s_w)
    grad_x_reshaped = relay.reshape(grad_x_permuted, (n, c_in, p_h * np_order * x_order_h, p_w * np_order * x_order_w))

    # Step 5: Slice to remove padding: (..., x_pad_h:x_pad_h + in_h, x_pad_w:x_pad_w + in_w)
    grad_x_sliced = relay.strided_slice(grad_x_reshaped,
                                        begin=[0, 0, x_pad_h, x_pad_w],
                                        end=[n, c_in, x_pad_h + in_h, x_pad_w + in_w])

    backward_data = grad_x_sliced

    backward_weight = grad_w
    # backward_weight = grad_w_conv
    
    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
        relay.zeros_like(o_order),
    ]
    return [
        backward_data,
        # relay.zeros_like(o_data),
        backward_weight,
        # relay.zeros_like(o_weight),
        backward_bias,
        # relay.zeros_like(o_bias),
        # backward_zero_x,
        relay.zeros_like(o_zx),
        # backward_zero_y,
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
        relay.zeros_like(o_order),
    ]

def sparse_in_channel_mcunetconv2davg_grad(orig, grad, topk=None):
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale, o_order = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale, order = new_inputs

    print("CHECK IN DATA TYPE", o_data.checked_type)
    print("CHECK IN WEIGHT TYPE", o_weight.checked_type)
    
    np_order = 2

    # Get the shape from the checked_type

    attrs = orig.attrs
    n, c_in, in_h, in_w = input_shape
    c_out, _, k_h, k_w = weight_shape
    stride_h, stride_w = get_const_tuple(attrs.strides)
    _, _, padding_h, padding_w = get_const_tuple(attrs.padding)
    d_h, d_w = get_const_tuple(attrs.dilation)

    # Calculate output height and width
    out_h = math.floor((in_h + 2 * padding_h - (d_h * (k_h - 1) + 1)) / stride_h + 1)
    out_w = math.floor((in_w + 2 * padding_w - (d_w * (k_w - 1) + 1)) / stride_w + 1)

    # Perform truncation

    p_h = ceil(in_h / np_order)
    p_w = ceil(in_w / np_order)
    
    print("p_h=", p_h)
    print("p_w=", p_w)

    # Sum weight over last two dimensions and clamp to avoid overflow
    weight_sum = relay.sum(weight, axis=[-1, -2])
    weight_sum = relay.nn.mcutruncate(weight_sum)
    weight_sum = relay.cast(weight_sum, "float32")
    # Calculate the output shape after summing over the last two dimensions
    weight_sum_shape = calculate_sum_shape(weight_shape, axes=[-1, -2])

    x_order_h = (np_order * stride_h)
    x_order_w = (np_order * stride_w)

    x_pad_h, x_pad_w = ceil((p_h * x_order_h - in_h) / 2), ceil((p_w * x_order_w - in_w) / 2)

    print("x_order_h=", x_order_h, "x_order_w=", x_order_w, "x_order_h=", x_order_h, "x_order_w", x_order_w, "x_pad_h=", x_pad_h, "x_pad_w", x_pad_w)

    #TODO: check strides bigger than 1

    # Average pooling with custom divisor
    x_avg = relay.nn.avg_pool2d(data, pool_size=(x_order_h, x_order_w), strides=(x_order_h, x_order_w),
                                padding=(x_pad_h, x_pad_w), ceil_mode=False, count_include_pad=False)

    _, _, x_avg_h, x_avg_w = calculate_pool_output_shape(input_shape, x_order_h, x_order_h, x_pad_h)

    # Multiply the result of average pooling by the kernel size to get the sum
    kernel_area = x_order_h * x_order_w
    x_sum = relay.multiply(x_avg, relay.const(kernel_area, dtype="float32"))

    x_sum_shape = calculate_pool_output_shape(input_shape, x_order_h, x_order_h, x_pad_h)
    # Clamp the result and round it
    x_sum = relay.nn.mcutruncate(x_sum, out_dtype="int8")
    x_sum = relay.cast(x_sum, "float32")

    output_shape = n, c_out, out_h, out_w

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # Compute padding for the gradient pooling operation

    gy_h = out_h
    gy_w = out_w

    grad_y_pad_h = ceil((p_h * np_order - gy_h) / 2)
    grad_y_pad_w = ceil((p_w * np_order - gy_w) / 2)


    # Average pooling for the gradient
    grad_y_avg = relay.nn.avg_pool2d(grad, pool_size=(np_order, np_order), strides=(np_order, np_order),
                                     padding=(grad_y_pad_h, grad_y_pad_w), count_include_pad=False)

    _, _, grad_y_avg_h, grad_y_avg_w = calculate_pool_output_shape(output_shape, np_order, np_order, grad_y_pad_h)


    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (1, 1)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = ( - 1) * stride_h - fpad_top - fpad_bottom + 1
    out_w = (grad_y_avg_w - 1) * stride_w - fpad_left - fpad_right + 1
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad_y_avg,
        weight_sum,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=ograd_y_avg_hutput_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(1, 1),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        assert attrs.groups == 1
        x_sum = relay.strided_slice(
            x_sum,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )

    grad_y_avg = tile(grad_y_avg, [1, tmp_inc // attrs.groups, 1, 1])
    grad_y_avg = reshape(grad_y_avg, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    x_sum = reshape(x_sum, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        x_sum,
        grad_y_avg,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_y_avg_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_y_avg_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_depth_wise_mcunetconv2davg_grad(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale, order = new_inputs

    print("CHECK IN DATA TYPE", o_data.checked_type)
    print("CHECK IN WEIGHT TYPE", o_weight.checked_type)
    
    np_order = 2

    # Get the shape from the checked_type

    attrs = orig.attrs
    n, c_in, in_h, in_w = input_shape
    c_out, _, k_h, k_w = weight_shape
    stride_h, stride_w = get_const_tuple(attrs.strides)
    _, _, padding_h, padding_w = get_const_tuple(attrs.padding)
    d_h, d_w = get_const_tuple(attrs.dilation)

    # Calculate output height and width
    out_h = math.floor((in_h + 2 * padding_h - (d_h * (k_h - 1) + 1)) / stride_h + 1)
    out_w = math.floor((in_w + 2 * padding_w - (d_w * (k_w - 1) + 1)) / stride_w + 1)

    # Perform truncation

    p_h = ceil(in_h / np_order)
    p_w = ceil(in_w / np_order)
    
    print("p_h=", p_h)
    print("p_w=", p_w)

    # Sum weight over last two dimensions and clamp to avoid overflow
    weight_sum = relay.sum(weight, axis=[-1, -2])
    weight_sum = relay.nn.mcutruncate(weight_sum)
    weight_sum = relay.cast(weight_sum, "float32")
    # Calculate the output shape after summing over the last two dimensions
    weight_sum_shape = calculate_sum_shape(weight_shape, axes=[-1, -2])

    x_order_h = (np_order * stride_h)
    x_order_w = (np_order * stride_w)

    x_pad_h, x_pad_w = ceil((p_h * x_order_h - in_h) / 2), ceil((p_w * x_order_w - in_w) / 2)

    print("x_order_h=", x_order_h, "x_order_w=", x_order_w, "x_order_h=", x_order_h, "x_order_w", x_order_w, "x_pad_h=", x_pad_h, "x_pad_w", x_pad_w)

    #TODO: check strides bigger than 1

    # Average pooling with custom divisor
    x_avg = relay.nn.avg_pool2d(data, pool_size=(x_order_h, x_order_w), strides=(x_order_h, x_order_w),
                                padding=(x_pad_h, x_pad_w), ceil_mode=False, count_include_pad=False)

    _, _, x_avg_h, x_avg_w = calculate_pool_output_shape(input_shape, x_order_h, x_order_h, x_pad_h)

    # Multiply the result of average pooling by the kernel size to get the sum
    kernel_area = x_order_h * x_order_w
    x_sum = relay.multiply(x_avg, relay.const(kernel_area, dtype="float32"))

    x_sum_shape = calculate_pool_output_shape(input_shape, x_order_h, x_order_h, x_pad_h)
    # Clamp the result and round it
    x_sum = relay.nn.mcutruncate(x_sum, out_dtype="int8")
    x_sum = relay.cast(x_sum, "float32")

    output_shape = n, c_out, out_h, out_w

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # Compute padding for the gradient pooling operation

    gy_h = out_h
    gy_w = out_w

    grad_y_pad_h = ceil((p_h * np_order - gy_h) / 2)
    grad_y_pad_w = ceil((p_w * np_order - gy_w) / 2)


    # Average pooling for the gradient
    grad_y_avg = relay.nn.avg_pool2d(grad, pool_size=(np_order, np_order), strides=(np_order, np_order),
                                     padding=(grad_y_pad_h, grad_y_pad_w), count_include_pad=False)

    _, _, grad_y_avg_h, grad_y_avg_w = calculate_pool_output_shape(output_shape, np_order, np_order, grad_y_pad_h)


    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (1, 1)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = ( - 1) * stride_h - fpad_top - fpad_bottom + 1
    out_w = (grad_y_avg_w - 1) * stride_w - fpad_left - fpad_right + 1
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    # Reshape weight_sum for broadcasting
    weight_sum_reshaped = relay.reshape(weight_sum, (1, c_out, 1, 1))

    # Compute grad_x_sum by multiplying grad_y_avg and weight_sum_reshaped
    grad_x_sum = grad_y_avg * weight_sum_reshaped



    # Reshape and permute gradient for data
    # Step 1: Reshape grad_x_sum to (n, c_in, p_h, p_w, 1, 1)
    grad_x = relay.reshape(grad_x_sum, (n, c_in, p_h, p_w, 1, 1))

    # Step 2: Tile grad_x to (n, c_in, p_h, p_w, order * s_h, order * s_w)
    # Calculate the repetition factor for the tile operation:
    tile_reps = (1, 1, 1, 1, np_order * x_order_h, np_order * x_order_w)
    grad_x_tiled = relay.tile(grad_x, reps=tile_reps)

    # Step 3: Permute axes (0, 1, 2, 4, 3, 5) -> equivalent to relay.transpose
    grad_x_permuted = relay.transpose(grad_x_tiled, axes=(0, 1, 2, 4, 3, 5))

    # Step 4: Reshape to (n, c_in, p_h * order * s_h, p_w * order * s_w)
    grad_x_reshaped = relay.reshape(grad_x_permuted, (n, c_in, p_h * np_order * x_order_h, p_w * np_order * x_order_w))

    # Step 5: Slice to remove padding: (..., x_pad_h:x_pad_h + in_h, x_pad_w:x_pad_w + in_w)
    grad_x_sliced = relay.strided_slice(grad_x_reshaped,
                                        begin=[0, 0, x_pad_h, x_pad_w],
                                        end=[n, c_in, x_pad_h + in_h, x_pad_w + in_w])

    backward_data = grad_x_sliced

    # backward_data = _nn.conv2d_transpose(
    #     grad,
    #     weight,
    #     strides=attrs.strides,
    #     padding=attrs.padding,
    #     dilation=attrs.dilation,
    #     groups=attrs.groups,
    #     output_padding=output_padding,
    #     # to fix codegen bug
    #     # TODO(lyken17): figure out why missing default value leads to error
    #     kernel_size=(filter_h, filter_w),
    #     channels=in_channel,
    # )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    groups = attrs.groups
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        tmp_ouc = round(topk * out_channel)
        x_sum = relay.strided_slice(
            x_sum,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, p_h, p_w]),
        )
        grad_y_avg = relay.strided_slice(
            grad_y_avg,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([grad_n, tmp_ouc, grad_y_avg_h, grad_y_avg_w]),
        )
        groups = tmp_inc

    # Compute grad_w_sum by multiplying grad_y_avg and x_sum
    grad_w_sum = grad_y_avg * x_sum

    # Sum grad_w_sum along axes [0, 2, 3] (batch, height, width)
    grad_w_sum = relay.sum(grad_w_sum, axis=[0, 2, 3])

    # Reshape grad_w_sum to (c_out, 1, 1, 1)
    grad_w_sum = relay.reshape(grad_w_sum, (c_out, 1, 1, 1))

    # Use relay.tile instead of broadcast_to
    # Calculate the repetition factor for tiling:
    tile_reps_w = (1, 1, k_h, k_w)  # Repeat across height (k_h) and width (k_w)
    grad_w = relay.tile(grad_w_sum, reps=tile_reps_w)

    backward_weight = grad_w
    # grad = tile(grad, [1, tmp_inc // groups, 1, 1])
    # grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    # data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    # backward_weight = _nn.conv2d(
    #     data,
    #     grad,
    #     strides=attrs.dilation,
    #     padding=attrs.padding,
    #     dilation=attrs.strides,
    #     groups=tmp_inc * batch,
    # )

    # # infer shape of backward_weight
    # padded_weight_grad_h = (
    #     in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    # ) // dilation_h + 1
    # padded_weight_grad_w = (
    #     in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    # ) // dilation_w + 1
    # backward_weight = reshape(
    #     backward_weight,
    #     [
    #         batch,
    #         tmp_inc // groups,
    #         tmp_ouc,
    #         padded_weight_grad_h,
    #         padded_weight_grad_w,
    #     ],
    # )

    # backward_weight = _sum(backward_weight, axis=0)
    # backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    # assert padded_weight_grad_h >= filter_h
    # assert padded_weight_grad_w >= filter_w
    # if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
    #     backward_weight = strided_slice(
    #         backward_weight,
    #         begin=[0, 0, 0, 0],
    #         end=[tmp_ouc, tmp_inc // groups, filter_h, filter_w],
    #     )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)
    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]

def sparse_mcunetconv2d_grad_tmp_fix(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    # if topk is not None:
    #     tmp_inc = round(topk * in_channel)
    #     tmp_ouc = round(topk * out_channel)
    #     data = relay.strided_slice(data,
    #         begin=relay.const([0, 0, 0, 0]),
    #         end=relay.const([batch, tmp_inc, in_h, in_w]),
    #     )
    #     grad = relay.strided_slice(grad,
    #         begin=relay.const([0, 0, 0, 0]),
    #         end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
    #     )

    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, in_channel // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask

    backward_weight = relay.strided_slice(
        backward_weight,
        begin=(0, 0, 0, 0),
        end=(round(topk * out_channel), in_channel, filter_h, filter_w),
    )
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_in_channel_mcunetconv2d_grad(orig, grad, topk=None):
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        assert attrs.groups == 1
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )

    grad = tile(grad, [1, tmp_inc // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_depth_wise_mcunetconv2d_grad(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    groups = attrs.groups
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        tmp_ouc = round(topk * out_channel)
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )
        grad = relay.strided_slice(
            grad,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
        )
        groups = tmp_inc

    grad = tile(grad, [1, tmp_inc // groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)
    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_mcunetconv2d_grad(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        tmp_ouc = round(topk * out_channel)
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )
        grad = relay.strided_slice(
            grad,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
        )

    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, in_channel // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


@register_gradient("nn.mcuadd")
def mcunetconv2d_grad(orig, grad):
    # cast to 32bits for backward computation
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs
    grad = relay.cast(grad, "float32")

    # grad_zero_y = grad_output.sum([0, 2, 3])
    grad_zero_y = relay.sum(grad)
    # grad_sum = grad_output / scale_y.item()
    new_scale_y = relay.reshape(scale_y, newshape=[1, -1, 1, 1])
    grad_sum = grad / new_scale_y

    # grad_x1 = grad_sum * scale_x1.item()
    new_scale_x1 = relay.reshape(scale_x1, newshape=[1, -1, 1, 1])
    grad_x1 = grad_sum * new_scale_x1

    # grad_x2 = grad_sum * scale_x2.item()
    new_scale_x2 = relay.reshape(scale_x2, newshape=[1, -1, 1, 1])
    grad_x2 = grad_sum * new_scale_x2

    # grad_zero_x1 = - grad_x1.sum([0, 2, 3])
    grad_zero_x1 = -relay.sum(grad_x1)
    # grad_zero_x2 = - grad_x2.sum([0, 2, 3])
    grad_zero_x2 = -relay.sum(grad_x2)
    return [
        grad_x1,
        grad_x2,
        grad_zero_x1,
        grad_zero_x2,
        relay.zeros_like(scale_x1),
        relay.zeros_like(scale_x2),
        grad_zero_y,
        relay.zeros_like(scale_y),
    ]


@register_gradient("nn.log_softmax", level=PROJECT_LEVEL)
def log_softmax_grad(orig, grad):
    """Gradient of log_softmax"""
    return [grad - _sum(grad, axis=orig.attrs.axis, keepdims=True) * exp(orig)]


@register_gradient("nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(orig, grad):
    x, y = orig.args
    # shape = shape_of(x)
    # batch_size = take(shape, const(0, dtype="int32"), axis=0)
    # print(x.checked_type.shape[0], type(x.checked_type.shape[0]))
    batch_size = const(int(x.checked_type.shape[0]))
    # print(batch_size)
    # input()
    grad = grad / batch_size.astype(x.checked_type.dtype)
    return [-grad * y, -grad * x]


# print(GRAD_OP_MAP.keys())
@register_gradient("nn.dense")
def dense_grad(orig, grad):
    x, w = orig.args
    # print("DEBUG", x.checked_type.shape, w.checked_type.shape, grad.checked_type.shape)
    # print("DEBUG dense_grad")
    dydx = relay.nn.matmul(grad, w)
    dydw = relay.nn.matmul(relay.transpose(grad), x)
    return [dydx, dydw]


@register_gradient("nn.bias_add")
def bias_add_grad(orig, grad):
    """Returns gradient of bias_add"""
    data = orig.args[0]
    return [
        # collapse_sum_like(grad, data),
        grad,
        _sum(grad, orig.attrs.axis, keepdims=False, exclude=True),
    ]


@register_gradient("clip")
def clip_grad(orig, grad):
    """Returns grad * (select(x < min || max < x , 0, 1))."""
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    zeros = zeros_like(x)
    ones = ones_like(x)
    # a_mins = broadcast_to_like(const(a_min, dtype=x.checked_type.dtype), x)
    # a_maxs = broadcast_to_like(const(a_max, dtype=x.checked_type.dtype), x)
    a_mins = relay.zeros(x.checked_type.shape, dtype=x.checked_type.dtype) * const(
        a_min
    )
    a_maxs = relay.ones(x.checked_type.shape, dtype=x.checked_type.dtype) * const(a_max)
    return [where(less(x, a_mins), zeros, where(less(a_maxs, x), zeros, ones * grad))]


@register_gradient("mean")
def mean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    if axis is None:
        axis = list(range(len(data.checked_type.concrete_shape)))
    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]
    # return [broadcast_to_like(grad * const(mult, dtype=data.checked_type.dtype), data)]
    return [grad * const(mult, dtype=data.checked_type.dtype)]


@register_gradient("nn.relu")
def relu_grad(orig, grad):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    zeros = relay.zeros_like(x)
    return [
        relay.op.transform.where(relay.less(x, zeros), zeros, grad),
    ]


@register_gradient("add")
def add_grad(orig, grad):
    """Returns [grad, grad]"""
    # TODO: check why collapse_sum is necessary here
    return [grad, grad]


@register_gradient("nn.adaptive_avg_pool2d", level=PROJECT_LEVEL + 1)
def adaptive_avg_pool2d_grad(orig, grad):
    # print(
    #     f"|adaptive_avg_pool2d_grad| (#num of args: {len(orig.args)}):",
    #     [(type(_), _.checked_type.shape, shape_of(_)) for _ in orig.args],
    # )
    """Returns the gradient of adaptive_avg_pool2d."""
    data = orig.args[0]
    shape = data.checked_type.shape
    attrs = orig.attrs  # ['output_size', 'layout', 'out_layout']
    layout = attrs.layout

    output_size = attrs.output_size
    assert layout in ["NCHW", "NHWC"], f"un-supported layout {layout}"
    if layout == "NCHW":
        pool_size = shape[2], shape[3]
    elif layout == "NHWC":
        pool_size = shape[1], shape[2]

    # TODO: fix the shape check
    pool_size = (pool_size[0] // output_size[0], pool_size[1] // output_size[1])

    pool_grad = _nn.avg_pool2d_grad(
        grad, data, pool_size=pool_size, strides=(1, 1), padding=(0, 0), layout=layout
    )
    # print(type(pool_grad), pool_grad)
    return [
        pool_grad,
    ]


@register_gradient("nn.conv2d", level=PROJECT_LEVEL + 1)
def conv2d_grad(orig, grad):
    """Gradient of conv2d"""
    attrs = orig.attrs
    data, weight = orig.args
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(weight.checked_type.shape)
    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(grad_h, grad_w),
        channels=batch * out_channel * in_channel // attrs.groups,
    )
    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    return [backward_data, backward_weight]


# @register_gradient("nn.conv2d", level=PROJECT_LEVEL+10)
# def conv2d_grad(orig, grad):
#     """Gradient of conv2d"""
#     attrs = orig.attrs
#     data, weight = orig.args
#     data_shape = get_const_tuple(data.checked_type.shape)
#     weight_shape = get_const_tuple(weight.checked_type.shape)
#     _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
#     _, _, in_h, in_w = data_shape
#     _, _, filter_h, filter_w = weight_shape

#     # infer output_padding
#     fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
#         get_const_tuple(attrs.padding), (filter_h, filter_w)
#     )
#     stride_h, stride_w = get_const_tuple(attrs.strides)
#     out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
#     out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
#     output_padding = (in_h - out_h, in_w - out_w)

#     assert attrs.data_layout == "NCHW", "only support NCHW data layout"
#     assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
#     assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

#     if attrs.out_dtype in ["", None]:
#         assert data.checked_type, "Call InferType first."
#         out_dtype = data.checked_type.dtype
#     else:
#         out_dtype = attrs.out_dtype

#     backward_data = _nn.conv2d_transpose(
#         grad,
#         weight,
#         strides=attrs.strides,
#         padding=attrs.padding,
#         dilation=attrs.dilation,
#         groups=attrs.groups,
#         output_padding=output_padding,
#         out_dtype=out_dtype,
#     )

#     backward_weight = _nn.conv2d_backward_weight(
#         grad,
#         data,
#         strides=attrs.strides,
#         padding=attrs.padding,
#         dilation=attrs.dilation,
#         groups=attrs.groups,
#         channels=attrs.channels,
#         kernel_size=(filter_h, filter_w),
#         grad_layout=attrs.out_layout if attrs.out_layout else attrs.data_layout,
#         data_layout=attrs.data_layout,
#         kernel_layout=attrs.kernel_layout,
#         out_dtype=out_dtype,
#     )

#     return [backward_data, backward_weight]
