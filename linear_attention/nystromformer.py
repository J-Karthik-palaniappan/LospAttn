import torch
import torch.nn.functional as F
from .utils import landmark_pool_2D, moore_penrose_iter_pinv

def nystromformer_torch(q, k, v, landmarks=32, iters=4, contain_cls=True):
    scale = (q.shape[-1])**-0.5
    q_m = landmark_pool_2D(q, landmarks, contain_cls=contain_cls)
    k_m = landmark_pool_2D(k, landmarks, contain_cls=contain_cls)
    
    kernel_1 = F.softmax(q @ k_m.transpose(-1, -2) * scale, dim=-1)
    kernel_2 = F.softmax(q_m @ k_m.transpose(-1, -2) * scale, dim=-1)
    kernel_3 = F.softmax(q_m @ k.transpose(-1, -2) * scale, dim=-1)
    out = kernel_1 @ (moore_penrose_iter_pinv(kernel_2, iters) @ (kernel_3 @ v))
    return out

# ==================================== Triton fused codes ===================================

from .triton_codes.fused_pinv import fused_softmax_pinv

def nystromformer(q, k, v, landmarks=32, iters=4, contain_cls=True):
    scale = (q.shape[-1])**-0.5
    q_m = landmark_pool_2D(q, landmarks, contain_cls=contain_cls)
    k_m = landmark_pool_2D(k, landmarks, contain_cls=contain_cls)

    out = F.scaled_dot_product_attention(q_m, k ,v, scale=scale)
    if landmarks > 64:
        kernel_2 = F.softmax(q_m @ k_m.transpose(-1, -2) * scale, dim=-1)
        out = moore_penrose_iter_pinv(kernel_2, iters) @ out
    else:
        out = fused_softmax_pinv(q_m, k_m, scale=scale, iters=iters) @ out
    out = F.scaled_dot_product_attention(q, k_m ,out, scale=scale)
    return out
