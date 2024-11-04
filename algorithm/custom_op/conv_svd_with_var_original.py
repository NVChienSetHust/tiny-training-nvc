import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn


###### SVD by choosing base on var -> old version -> dont use
def find_k(X, var=0.9):
    # Tính toán SVD
    U, S, Vt = th.linalg.svd(X)

    # Tính toán tổng phương sai
    total_variance = th.sum(S**2, dim=2)
    cum_sum_tensor = th.cumsum(S**2, dim=2)

    # Broadcast tensor total_variance để phù hợp với kích thước của cum_sum_tensor
    total_variance_broadcasted = total_variance.unsqueeze(2)

    # Thực hiện phép chia element-wise
    explained_variance = cum_sum_tensor / total_variance_broadcasted

    bool_indices = th.ge(explained_variance, var)
    k = th.argmax(bool_indices.type(th.int), dim=-1) # argmax sẽ trả về chỉ số của giá trị đầu tiên nếu có nhiều giá trị lớn nhất (Nhiều giá trị True), và trong trường hợp này, giá trị 1 (True) sẽ được ưu tiên trước giá trị 0 (False).
    k = k + 1 # k chỉ là index, muốn dùng để cắt tensor thì phải + 1
    return U, S, Vt, k

def find_Sk_mask(S, k): # Dựa vào k, trả về mask chứa các phần tử cần thiết trong S và số lượng của chúng
    # Tạo tensor chứa chỉ số để chọn
    indices = (th.arange(S.shape[2]).expand(S.shape[0], S.shape[1], -1)).to(S.device)

    # Tạo mask để chỉ giữ lại các phần tử đầu tiên
    Sk_mask = indices < k.unsqueeze(2)
    Sk_num_elem = int(Sk_mask.sum())
    return Sk_mask, Sk_num_elem

def find_Uk_mask(U, k): # Dựa vào k, trả về mask chứa các phần tử cần thiết trong U và số lượng của chúng
    # Tạo mask để chỉ giữ lại k cột đầu tiên
    Uk_mask = (th.arange(U.shape[2]).unsqueeze(0).unsqueeze(0)).to(U.device) < k.unsqueeze(2).unsqueeze(3)
    Uk_num_elem = int(Uk_mask.sum())*U.shape[2]
    return Uk_mask, Uk_num_elem


def find_Vk_mask(Vt, k): # Dựa vào k, trả về mask chứa các phần tử cần thiết trong Vt và số lượng của chúng
    V = Vt.permute(0, 1, 3, 2) # Transpose các ma trận trong Vt thành V
    # Tạo mask để chỉ giữ lại k cột đầu tiên
    Vk_mask = (th.arange(V.shape[2]).unsqueeze(0).unsqueeze(0)).to(Vt.device) < k.unsqueeze(2).unsqueeze(3)
    Vk_num_elem = int(Vk_mask.sum())*Vt.shape[2]
    return Vk_mask, Vk_num_elem # Đây là mask của ma trận Vk

def extract_values_with_mask_Sk(S, mask_Sk): # Chỉ trích xuất ra các giá trị cần thiết từ S dựa trên mask
    # Lấy các giá trị mà mask chỉ định từ tensor S
    Sk_raw = S[mask_Sk]
    return Sk_raw

def restore_with_mask_Sk(Sk_raw, mask_Sk): # Khôi phục lại Sk thành tensor có shape giống S dựa trên đầu ra của extract_values_with_mask_Sk, các phần tử bị thiếu sẽ cho bằng 0
    # Khôi phục lại tensor Sk từ Sk_raw và mask
    Sk = th.zeros(mask_Sk.shape).to(Sk_raw.device)
    Sk[mask_Sk] = Sk_raw
    return Sk

def extract_values_with_mask_Uk(U, mask_Uk):
    selected_columns = th.nonzero(mask_Uk, as_tuple=False).to(U.device)
    Uk_raw = U[selected_columns[:, 0], selected_columns[:, 1], :, selected_columns[:, 3]]
    return Uk_raw


def restore_with_mask_Uk(Uk_raw, mask_Uk):
    Uk = th.zeros(mask_Uk.shape[0], mask_Uk.shape[1], mask_Uk.shape[3], mask_Uk.shape[3]).to(Uk_raw.device) # Khôi phục thành tensor có shape giống U
    selected_columns = th.nonzero(mask_Uk, as_tuple=False)
    Uk[selected_columns[:, 0], selected_columns[:, 1], :, selected_columns[:, 3]] = Uk_raw
    return Uk

###############################################################
class Conv2dSVDop_with_var(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups) # Chỗ này như bình thường

        ### Kiểu svd dựa trên variance
        U, S, Vt, k = find_k(input, var=var)
        mask_Uk, Uk_num_elem = find_Uk_mask(U, k)
        mask_Sk, Sk_num_elem = find_Sk_mask(S, k)
        mask_Vk, Vk_t_num_elem = find_Vk_mask(Vt, k)
        # print("this is k: ", k)
        Sk_raw = extract_values_with_mask_Sk(S, mask_Sk)
        Uk_raw = extract_values_with_mask_Uk(U, mask_Uk)
        Vk_raw = extract_values_with_mask_Uk(Vt.permute(0, 1, 3, 2), mask_Vk)
        k = th.max(k)
        ctx.save_for_backward(mask_Uk, mask_Sk, mask_Vk, Sk_raw, Uk_raw, Vk_raw, k, weight, bias)



        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        ### Kiểu svd dựa trên variance
        mask_Uk, mask_Sk, mask_Vk, Sk_raw, Uk_raw, Vk_raw, k, weight, bias = ctx.saved_tensors
        Sk = restore_with_mask_Sk(Sk_raw, mask_Sk)
        Uk = restore_with_mask_Uk(Uk_raw, mask_Uk)
        Vk_restored = restore_with_mask_Uk(Vk_raw, mask_Vk)
        Vk_t = Vk_restored.permute(0, 1, 3, 2)

        Uk = Uk[:,:,:,:k]
        Sk = Sk[:,:,:k]
        Vk_t = Vk_t[:,:,:k,:]
        
        input = th.matmul(th.matmul(Uk, th.diag_embed(Sk)), Vk_t)



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
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None # Trả về gradient ứng với cái arg ở forward


class Conv2dSVD_with_var(nn.Conv2d):
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
        super(Conv2dSVD_with_var, self).__init__(in_channels=in_channels,
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
            y = Conv2dSVDop_with_var.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var)
        else:
            y = super().forward(x)
        return y

def wrap_convSVD_with_var_layer(conv, SVD_var, active):
    new_conv = Conv2dSVD_with_var(in_channels=conv.in_channels,
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