
import torch
import triton
import triton.language as tl


@triton.jit
def fused_final_attention_kernel(
    Q, M, U, DEN, O,
    sqb, sqh, sqn, sqd,
    smb, smh, sms, smd,
    sub, suh, sus, sud,
    sdb, sdh, sds,
    H: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BQ: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    FAST_FP16: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    qi = pid_q * BQ + tl.arange(0, BQ)
    d = tl.arange(0, BD)

    vq = qi < N
    vd = d < D

    b = pid_bh // H
    h = pid_bh % H

    qb = Q + b * sqb + h * sqh
    mb = M + b * smb + h * smh
    ub = U + b * sub + h * suh
    db = DEN + b * sdb + h * sdh

    q = tl.load(
        qb + qi[:, None] * sqn + d[None, :] * sqd,
        vq[:, None] & vd[None, :],
        0
    )

    # Online softmax accumulators.
    m_i = tl.full((BQ,), -float("inf"), tl.float32)
    l_i = tl.zeros((BQ,), tl.float32)

    acc_u = tl.zeros((BQ, BD), tl.float32)
    acc_den = tl.zeros((BQ,), tl.float32)

    for start in range(tl.cdiv(S, BS)):

        s = start * BS + tl.arange(0, BS)
        vs = s < S

        mt = tl.load(
            mb + d[:, None] * smd + s[None, :] * sms,
            vd[:, None] & vs[None, :],
            0
        )

        # QK score.
        if FAST_FP16:
            scores = (
                tl.dot(q, mt) * SCALE
            ).to(tl.float16).to(tl.float32)
        else:
            scores = tl.dot(
                q, mt,
                input_precision="tf32"
            ) * SCALE

        scores = tl.where(
            vs[None, :],
            scores,
            -float("inf")
        )

        # Online softmax.
        m_chunk = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_chunk)

        alpha = tl.exp(m_i - m_new)

        p = tl.exp(scores - m_new[:, None])
        p = tl.where(vs[None, :], p, 0.0)

        l_i = alpha * l_i + tl.sum(p, axis=1)

        if FAST_FP16:
            p_dot = p.to(tl.float16)
        else:
            p_dot = p

        # Load U.
        u = tl.load(
            ub + s[:, None] * sus + d[None, :] * sud,
            vs[:, None] & vd[None, :],
            0
        )

        # Weighted numerator.
        if FAST_FP16:
            chunk_u = tl.dot(
                p_dot, u
            ).to(tl.float32)
        else:
            chunk_u = tl.dot(
                p_dot, u,
                input_precision="tf32"
            )

        acc_u = alpha[:, None] * acc_u + chunk_u

        # Weighted denominator.
        den = tl.load(
            db + s * sds,
            vs,
            0
        ).to(tl.float32)

        acc_den = (
            alpha * acc_den
            + tl.sum(p * den[None, :], axis=1)
        )

        m_i = m_new

    # Original normalized-softmax denominator semantics.
    result = acc_u / (
        acc_den[:, None] + EPS * l_i[:, None]
    )

    op = O + pid_bh * (N * D) + qi[:, None] * D + d[None, :]

    tl.store(
        op,
        result,
        vq[:, None] & vd[None, :]
    )


def fused_final_attention(
    q, m, u, denom,
    scale=None,
    eps=1e-12,
    block_q=16,
    block_s=32,
):
    B, H, N, D = q.shape
    S = m.shape[2]

    assert m.shape == u.shape == (B, H, S, D)
    assert denom.shape == (B, H, S, 1)

    assert q.dtype == m.dtype == u.dtype == denom.dtype
    assert q.dtype in (torch.float16, torch.float32)

    assert q.is_cuda and m.is_cuda and u.is_cuda and denom.is_cuda
    assert q.device == m.device == u.device == denom.device

    scale = D ** -0.5 if scale is None else scale

    out = torch.empty(
        (B, H, N, D),
        device=q.device,
        dtype=q.dtype
    )

    BQ = block_q
    BS = block_s
    BD = max(32, triton.next_power_of_2(D))

    fused_final_attention_kernel[
        (triton.cdiv(N, BQ), B * H)
    ](
        q, m, u, denom, out,

        *q.stride(),
        *m.stride(),
        *u.stride(),

        denom.stride(0),
        denom.stride(1),
        denom.stride(2),

        H=H,
        N=N,
        S=S,
        D=D,

        SCALE=scale,
        EPS=eps,

        BQ=BQ,
        BS=BS,
        BD=BD,

        FAST_FP16=(q.dtype == torch.float16),

        num_warps=4,
        num_stages=1,
        enable_fp_fusion=True,
    )

    return out
