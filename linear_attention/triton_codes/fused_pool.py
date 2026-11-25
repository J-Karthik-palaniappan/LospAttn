
import torch
import triton
import triton.language as tl
import math


def closest_square_factors(N):
    s = int(math.isqrt(N))
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i


@triton.jit
def _pool_qk_concat(
    Q, K, OUT,
    QSB: tl.constexpr, QSH: tl.constexpr, QSN: tl.constexpr, QSD: tl.constexpr,
    KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
    IH: tl.constexpr, IW: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    CLS_OFFSET: tl.constexpr,
    REGION_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    landmark = tl.program_id(1)

    b = bh // H
    h = bh % H

    d = tl.arange(0, BLOCK_D)

    q_base = Q + b * QSB + h * QSH
    k_base = K + b * KSB + h * KSH

    # Q occupies [0, M), K occupies [M, 2M)
    q_out = OUT + (bh * (2 * M) + landmark) * D
    k_out = OUT + (bh * (2 * M) + M + landmark) * D

    if landmark < CLS_OFFSET:

        q_cls = tl.load(
            q_base + d * QSD,
            d < D,
            other=0
        )

        k_cls = tl.load(
            k_base + d * KSD,
            d < D,
            other=0
        )

        tl.store(q_out + d, q_cls, d < D)
        tl.store(k_out + d, k_cls, d < D)

    else:

        spatial = landmark - CLS_OFFSET

        oh = spatial // OW
        ow = spatial % OW

        hs = oh * IH // OH
        he = ((oh + 1) * IH + OH - 1) // OH

        ws = ow * IW // OW
        we = ((ow + 1) * IW + OW - 1) // OW

        r = tl.arange(0, BLOCK_R)

        row = r // REGION_W
        col = r % REGION_W

        valid = (row < he - hs) & (col < we - ws)

        token = CLS_OFFSET + (hs + row) * IW + ws + col

        mask = valid[:, None] & (d[None, :] < D)

        q_vals = tl.load(
            q_base + token[:, None] * QSN + d[None, :] * QSD,
            mask,
            other=0
        ).to(tl.float32)

        k_vals = tl.load(
            k_base + token[:, None] * KSN + d[None, :] * KSD,
            mask,
            other=0
        ).to(tl.float32)

        area = (he - hs) * (we - ws)

        q_mean = tl.sum(q_vals, axis=0) / area
        k_mean = tl.sum(k_vals, axis=0) / area

        tl.store(q_out + d, q_mean, d < D)
        tl.store(k_out + d, k_mean, d < D)


def landmark_pool_2d_qk_concat(
    q,
    k,
    num_landmarks,
    contains_cls=False
):
    """
    Fused Q/K landmark pooling.

    Returns:
        out: contiguous [B, H, 2*M, D]

        out[:, :, :M, :] = pooled Q
        out[:, :, M:, :] = pooled K
    """

    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("q and k must have matching [B,H,N,D] shapes")

    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")

    if q.dtype != k.dtype or q.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32
    ):
        raise TypeError("Unsupported dtype")

    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        raise RuntimeError("Forward-only operation; use torch.no_grad()")

    b, h, n, d = q.shape

    spatial_n = n - int(contains_cls)
    spatial_m = num_landmarks - int(contains_cls)

    if spatial_n <= 0 or spatial_m <= 0 or d <= 0:
        raise ValueError("Invalid spatial tokens, landmarks, or D")

    ih, iw = closest_square_factors(spatial_n)
    oh, ow = closest_square_factors(spatial_m)

    region_h = max(
        ((i + 1) * ih + oh - 1) // oh - i * ih // oh
        for i in range(oh)
    )

    region_w = max(
        ((i + 1) * iw + ow - 1) // ow - i * iw // ow
        for i in range(ow)
    )

    block_r = triton.next_power_of_2(region_h * region_w)
    block_d = triton.next_power_of_2(d)

    # Single output allocation
    out = torch.empty(
        (b, h, 2 * num_landmarks, d),
        device=q.device,
        dtype=q.dtype
    )

    _pool_qk_concat[(b * h, num_landmarks)](
        q, k, out,
        *q.stride(),
        *k.stride(),
        h, d, num_landmarks,
        ih, iw, oh, ow,
        int(contains_cls),
        region_w,
        block_r,
        block_d,
        num_warps=4,
    )

    return out
