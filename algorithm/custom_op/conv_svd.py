import torch as torch
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn

###### SVD by choosing the first principle components

def truncated_svd(X, dim=0, k=1, driver=None):
    # dim is the dimension along which SVD will be applied
    n_samples, n_features = torch.prod(torch.tensor(X.shape[:dim+1])), torch.prod(torch.tensor(X.shape[dim+1:]))
    X_reshaped = X.reshape(n_samples, n_features)
    U, S, Vt = torch.linalg.svd(X_reshaped, full_matrices=False, driver=driver)
    
    # Take the first principal component
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    return torch.matmul(U_k, torch.diag_embed(S_k)), Vt_k
# def calculate_error(Sk, S):
#     Sk_ = th.reshape(Sk, (Sk.shape[0]*Sk.shape[1], Sk.shape[2]))
#     S_ = th.reshape(S, (S.shape[0]*S.shape[1], S.shape[2]))
#     error = 0
#     for i in range(len(Sk_)):
#         error += th.sum(Sk_[i]**2)/th.sum(S_[i]**2)
#     return error/i

def restore_tensor(Uk_Sk, Vk_t, shape):
    reconstructed_matrix = torch.matmul(Uk_Sk, Vk_t)
    shape = tuple(shape)
    return reconstructed_matrix.view(shape)

###############################################################
class Conv2dSVDop(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, k = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        input_Uk_Sk, input_Vk_t = truncated_svd(input, k=k)
        ctx.save_for_backward(input_Uk_Sk, input_Vk_t, torch.tensor(input.shape), weight, bias)

        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        input_Uk_Sk, input_Vk_t, input_shape, weight, bias = ctx.saved_tensors
        input = restore_tensor(input_Uk_Sk, input_Vk_t, input_shape)


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
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class Conv2dSVD(nn.Conv2d):
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
            k=1
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dSVD, self).__init__(in_channels=in_channels,
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
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2dSVDop.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.k)
        else:
            y = super().forward(x)
        return y

def wrap_convSVD_layer(conv, truncated_SVD_k, active):
    new_conv = Conv2dSVD(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         k=truncated_SVD_k # Truncated k
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv