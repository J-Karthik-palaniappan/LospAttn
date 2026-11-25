import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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


@triton.jit
def fused_error_reduce_kernel(
    EXACT,
    APPROX,
    ERR_OUT,
    REF_OUT,
    WIN_OUT,

    N: tl.constexpr,
    C: tl.constexpr,
    START: tl.constexpr,
    ROWS: tl.constexpr,

    WINDOW: tl.constexpr,
    STORE_BASE: tl.constexpr,

    BR: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)

    r = pid * BR + tl.arange(0, BR)
    j = tl.arange(0, BN)

    row_mask = r < ROWS
    col_mask = j < N

    e = tl.load(
        EXACT + r[:, None] * N + j[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.
    ).to(tl.float32)

    a = tl.load(
        APPROX + r[:, None] * N + j[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.
    ).to(tl.float32)

    # Normalize approximation
    a = a / tl.maximum(
        tl.sum(a, axis=1)[:, None], 1.e-12
    )

    # Landmark approximation error
    diff = a - e
    err_sq = tl.sum(diff * diff, axis=1)

    bh = r // C
    local_q = r % C
    pos = START + local_q

    out_idx = bh * N + pos

    tl.store(ERR_OUT + out_idx, err_sq, mask=row_mask)

    # Exact attention norm and window sparsification error
    if STORE_BASE:

        ref_sq = tl.sum(e * e, axis=1)

        # Boundary-adjusted fixed-width window
        wl = tl.maximum(pos - WINDOW // 2, 0)
        wr = tl.minimum(wl + WINDOW, N)
        wl = tl.maximum(wr - WINDOW, 0)

        inside = (
            (j[None, :] >= wl[:, None]) &
            (j[None, :] < wr[:, None])
        )

        outside_sq = tl.sum(
            tl.where(inside, 0., e * e),
            axis=1
        )

        tl.store(REF_OUT + out_idx, ref_sq, mask=row_mask)
        tl.store(WIN_OUT + out_idx, outside_sq, mask=row_mask)


@torch.inference_mode()
def approx_attention_error_triton(
    q,
    k,
    landmark_sizes=(32, 64, 128),
    iters=6,
    contain_cls=False,
    chunk_size=128,
    reduction_rows=1,
    window=512,
):
    B, H, N, D = q.shape
    BH = B * H
    scale = D ** -0.5

    q = q.to(device="cuda", dtype=torch.float16).contiguous()
    k = k.to(device=q.device, dtype=torch.float16).contiguous()
    kT = k.transpose(-2, -1)

    # ------------------------------------------------
    # Precompute landmark transformations
    # ------------------------------------------------
    precomputed = {}

    for L in landmark_sizes:
        qm = landmark_pool_2D(q, L, contain_cls)
        km = landmark_pool_2D(k, L, contain_cls)

        m = torch.cat((qm, km), dim=2)
        mT = m.transpose(-2, -1).contiguous()

        sftB = torch.softmax((m @ mT) * scale, dim=-1)
        sftC = torch.softmax((m @ kT) * scale, dim=-1)

        invB = moore_penrose_iter_pinv(sftB.float(), iters)

        T = (invB @ sftC.float()).to(torch.float16).contiguous()

        precomputed[L] = (mT, T)

    # ------------------------------------------------
    # Output storage
    # ------------------------------------------------
    errors_sq = {
        L: torch.empty(
            BH, N, device=q.device, dtype=torch.float32
        )
        for L in landmark_sizes
    }

    ref_sq = torch.empty(
        BH, N, device=q.device, dtype=torch.float32
    )

    win_sq = torch.empty_like(ref_sq)

    BN = triton.next_power_of_2(N)

    # ------------------------------------------------
    # Chunked attention
    # ------------------------------------------------
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        C = end - start

        q_chunk = q[:, :, start:end, :]

        exact = torch.softmax(
            (q_chunk @ kT) * scale,
            dim=-1
        ).contiguous()

        ROWS = BH * C

        for idx, L in enumerate(landmark_sizes):
            mT, T = precomputed[L]

            sftA = torch.softmax(
                (q_chunk @ mT) * scale,
                dim=-1
            )

            approx = (sftA @ T).contiguous()

            fused_error_reduce_kernel[
                (triton.cdiv(ROWS, reduction_rows),)
            ](
                exact,
                approx,
                errors_sq[L],
                ref_sq,
                win_sq,

                N,
                C,
                start,
                ROWS,

                window,
                idx == 0,

                reduction_rows,
                BN,

                num_warps=4,
                num_stages=1,
            )

        del exact, q_chunk

    # ------------------------------------------------
    # Final relative Frobenius errors
    # ------------------------------------------------
    ref = ref_sq.sum(dim=-1).reshape(B, H).sqrt().clamp_min(1e-12)

    results = {}

    for L in landmark_sizes:
        err = errors_sq[L].sum(dim=-1).reshape(B, H).sqrt()
        results[L] = err / ref

    # Window sparsification error (independent of L)
    win = win_sq.sum(dim=-1).reshape(B, H).sqrt()
    window_error = win / ref

    return results, window_error
