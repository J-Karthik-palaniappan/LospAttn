import torch
import torch.nn.functional as F
from .utils import landmark_pool_2D, moore_penrose_iter_pinv

def symmetric_nystrom_torch(q, k, v, landmarks=32, iters=4, contain_cls=False, return_attn=False):
    scale = (q.shape[-1])**-0.5
    q_m = landmark_pool_2D(q, landmarks, contain_cls=contain_cls) 
    k_m = landmark_pool_2D(k, landmarks, contain_cls=contain_cls) 
    m = torch.cat([q_m, k_m], dim=2)

    sftA = F.softmax(q @ m.transpose(-2, -1) * scale, dim=-1)
    sftB = F.softmax(m @ m.transpose(-2, -1) * scale, dim=-1)
    sftC = F.softmax(m @ k.transpose(-2, -1) * scale, dim=-1)

    invB = moore_penrose_iter_pinv(sftB, iters)

    v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    prod = sftA @ (invB @ (sftC @ v_aug))
    out = (prod[..., :-1] / (prod[..., -1].unsqueeze(-1) + 1e-12))

    if return_attn:
        prod = sftA @ invB @ sftC
        # prod = prod.clamp_min(0)
        attn = prod / (prod.sum(dim=-1, keepdims=True) + 1e-12)
        return attn, out

    return out

# ==================================== Triton fused codes ===================================

from .triton_codes.fused_pinv import fused_softmax_pinv
from .triton_codes.fused_symmattn import fused_final_attention
from .triton_codes.fused_pool import landmark_pool_2d_qk_concat


def symmetric_nystrom(q, k, v, landmarks=32, iters=4, contain_cls=True):
    scale = (q.shape[-1])**-0.5
    dtype = q.dtype
    m = landmark_pool_2d_qk_concat(q, k, landmarks, contains_cls=contain_cls)

    out = F.scaled_dot_product_attention(m, k ,v, scale=scale)

    threshold = 32 if dtype == torch.float32 else 64
    if landmarks > threshold:
        kernel_2 = F.softmax(m @ m.transpose(-1, -2) * scale, dim=-1)
        P = moore_penrose_iter_pinv(kernel_2, iters=iters)
    else:
        P = fused_softmax_pinv(m, m, scale=scale, iters=iters)

    out = P @ out
    denom = P.sum(dim=-1, keepdim=True)

    out = fused_final_attention(q, m.to(dtype=dtype), out.to(dtype=dtype), denom.to(dtype=dtype), scale=scale, eps=1e-12)
    return out
