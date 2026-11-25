import torch
import math
import torch.nn.functional as F
from einops import rearrange

# Inverse
def moore_penrose_iter_pinv(x, iters=5):
    abs_x = torch.abs(x)
    col   = abs_x.sum(dim=-1)
    row   = abs_x.sum(dim=-2)
    col_max = col.amax(dim=-1, keepdim=True)
    row_max = row.amax(dim=-1, keepdim=True)

    M = x.shape[-1]
    z = rearrange(x, "... i j -> ... j i") / (
        col_max.unsqueeze(-1) * row_max.unsqueeze(-2) #* M + 1e-15
    )
    
    I = torch.eye(x.shape[-2], device=x.device, dtype=x.dtype)
    I = rearrange(I, "i j -> () i j")
    for _ in range(iters):
        xz = x @ z
        z  = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z
    
# Pooling
def closest_square_factors(N):
    s = int(math.isqrt(N))   
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i

def landmark_pool_2D(q, num_landmarks, contain_cls=False):
    B, H, N, Dh = q.shape
    
    if contain_cls:
        # returns (num_landmarks-1) + 1 cls
        h, w = closest_square_factors(num_landmarks - 1)
        w1, w2 = closest_square_factors(N-1)
        q_cls = q[:, :, :1, :]  
        q_rest = q[:, :, 1:, :]
        q_m = F.adaptive_avg_pool2d(
            q_rest.reshape(B * H, w1, w2, Dh).permute(0, 3, 1, 2),
            output_size=(h, w)
        ).permute(0, 2, 3, 1).reshape(B, H, num_landmarks - 1, Dh)
        q_m = torch.cat([q_cls, q_m], dim=2)

    else:
        h, w = closest_square_factors(num_landmarks)
        w1, w2 = closest_square_factors(N)
        q_m = F.adaptive_avg_pool2d(
            q.reshape(B * H, w1, w2, Dh).permute(0, 3, 1, 2),
            output_size=(h, w)
        ).permute(0, 2, 3, 1).reshape(B, H, num_landmarks, Dh)

    return q_m
