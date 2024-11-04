import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, avg_pool2d
import torch.nn as nn
from math import ceil
# import os
# import psutil
# from util import get_memory_usage

###### HOSVD base on variance #############

def unfolding(n, A):
    shape = A.shape
    size = th.prod(th.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)

def truncated_svd(X, var=0.9, driver=None):
    # X is 2D tensor
    U, S, Vt = th.linalg.svd(X, full_matrices=False, driver=driver)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # k = (explained_variance >= var).nonzero()[0].item() + 1
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        # Nếu có ít nhất một phần tử >= var
        k = nonzero_indices[0].item() + 1
    else:
        # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
        k = explained_variance.argmax().item() + 1
    return U[:, :k], S[:k], Vt[:k, :]

def modalsvd(n, A, var, driver):
    nA = unfolding(n, A)
    # return torch.svd(nA)
    return truncated_svd(nA, var, driver)

def hosvd(A, var=0.9, driver=None):
    S = A.clone()
    
    u0, _, _ = modalsvd(0, A, var, driver)
    S = th.tensordot(S, u0, dims=([0], [0]))

    u1, _, _ = modalsvd(1, A, var, driver)
    S = th.tensordot(S, u1, dims=([0], [0]))

    u2, _, _ = modalsvd(2, A, var, driver)
    S = th.tensordot(S, u2, dims=([0], [0]))

    u3, _, _ = modalsvd(3, A, var, driver)
    S = th.tensordot(S, u3, dims=([0], [0]))
    return S, u0, u1, u2, u3

def restore_hosvd(S, u0, u1, u2, u3):
    # Initialize the restored tensor
    restored_tensor = S.clone()

    # Multiply each mode of the restored tensor by the corresponding U matrix
    restored_tensor = th.tensordot(restored_tensor, u0.t(), dims=([0], [0]))
    restored_tensor = th.tensordot(restored_tensor, u1.t(), dims=([0], [0]))
    restored_tensor = th.tensordot(restored_tensor, u2.t(), dims=([0], [0]))
    restored_tensor = th.tensordot(restored_tensor, u3.t(), dims=([0], [0]))
    return restored_tensor

###############################################################
class Conv2dHOSVDop_with_var(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        

        # idd = os.getpid()
        # process_ = psutil.Process(idd)

        input, weight, bias, stride, dilation, padding, groups, var = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)


        S, u0, u1, u2, u3 = hosvd(input, var=var)
        ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias)

        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        # print(f"Memory usage for forward: {get_memory_usage(process_) / (1024 ** 2)} MB")


        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        # pid = os.getpid()
        # process = psutil.Process(pid)

        S, u0, u1, u2, u3, weight, bias = ctx.saved_tensors
        input = restore_hosvd(S, u0, u1, u2, u3)

         
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs


        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        # print(f"Memory usage for backward: {get_memory_usage(process) / (1024 ** 2)} MB")
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class Conv2dHOSVD_with_var(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            var=1
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dHOSVD_with_var, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.var = var

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2dHOSVDop_with_var.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var)
        else:
            y = super().forward(x)
        return y

def wrap_convHOSVD_with_var_layer(conv, SVD_var, active):
    new_conv = Conv2dHOSVD_with_var(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv