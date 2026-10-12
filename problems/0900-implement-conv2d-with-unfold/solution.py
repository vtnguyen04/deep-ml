import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    # TODO: implement conv2d using F.unfold + matmul
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape

    pad_h = padding[0] if isinstance(padding, tuple) else padding
    pad_w = padding[1] if isinstance(padding, tuple) else padding
    stride_h = stride[0] if isinstance(stride, tuple) else stride
    stride_w = stride[1] if isinstance(stride, tuple) else stride

    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kW) // stride_w + 1

    x_unfolded = F.unfold(x, kernel_size=(kH, kW), stride=stride, padding=padding)

    W_flat = W.view(C_out, -1)

    out = W_flat @ x_unfolded

    if b is not None:
        out = out + b.view(1, -1, 1)

    out = out.view(N, C_out, H_out, W_out)

    return out







