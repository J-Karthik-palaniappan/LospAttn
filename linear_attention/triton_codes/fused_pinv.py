import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_pinv_kernel(
    q_ptr, k_ptr, out_ptr,
    qsb, qsm, qsd, ksb, ksm, ksd, osb, osm, osn,
    M: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BD: tl.constexpr,
    ITERS: tl.constexpr, SCALE: tl.constexpr,
    FAST_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.arange(0, BM)
    d = tl.arange(0, BD)
    valid_m = m < M
    valid_d = d < D
    mm = valid_m[:, None] & valid_m[None, :]

    qp = q_ptr + pid*qsb + m[:, None]*qsm + d[None, :]*qsd
    kp = k_ptr + pid*ksb + m[:, None]*ksm + d[None, :]*ksd

    Q = tl.load(qp, valid_m[:, None] & valid_d[None, :], 0)
    K = tl.load(kp, valid_m[:, None] & valid_d[None, :], 0)

    if FAST_FP16:
        S = (tl.dot(Q, tl.trans(K)) * SCALE).to(tl.float16)
    else:
        S = tl.dot(Q, tl.trans(K), input_precision="tf32") * SCALE

    Sm = tl.where(mm, S, -float("inf"))
    row_max = tl.max(Sm, axis=1)
    row_max = tl.where(valid_m, row_max, 0.0)
    P = tl.exp(S - row_max[:, None])
    P = tl.where(mm, P, 0.0)

    if FAST_FP16:
        P = P.to(tl.float16)
        row_sum = tl.sum(P.to(tl.float32), axis=1)
        X = (P.to(tl.float32) / row_sum[:, None]).to(tl.float16)
    else:
        row_sum = tl.sum(P, axis=1)
        X = P / row_sum[:, None]

    X = tl.where(mm, X, 0.0)

    row_l1 = tl.sum(tl.abs(X).to(tl.float32), axis=1)
    col_l1 = tl.sum(tl.abs(X).to(tl.float32), axis=0)
    row_norm = tl.max(tl.where(valid_m, row_l1, 0.0), axis=0)
    col_norm = tl.max(tl.where(valid_m, col_l1, 0.0), axis=0)

    if FAST_FP16:
        Z = (tl.trans(X).to(tl.float32) / (row_norm * col_norm)).to(tl.float16)
        I = ((m[:, None] == m[None, :]) & mm).to(tl.float16)
        for _ in range(ITERS):
            XZ = tl.dot(X, Z).to(tl.float16)
            A = tl.dot(XZ, (7.0 * I - XZ).to(tl.float16)).to(tl.float16)
            B = tl.dot(XZ, (15.0 * I - A).to(tl.float16)).to(tl.float16)
            Z = (0.25 * tl.dot(Z, (13.0 * I - B).to(tl.float16))).to(tl.float16)
    else:
        Z = tl.trans(X) / (row_norm * col_norm)
        I = ((m[:, None] == m[None, :]) & mm).to(tl.float32)
        for _ in range(ITERS):
            XZ = tl.dot(X, Z, input_precision="tf32")
            A = tl.dot(XZ, 7.0 * I - XZ, input_precision="tf32")
            B = tl.dot(XZ, 15.0 * I - A, input_precision="tf32")
            Z = 0.25 * tl.dot(Z, 13.0 * I - B, input_precision="tf32")

    op = out_ptr + pid*osb + m[:, None]*osm + m[None, :]*osn
    tl.store(op, Z, mm)


def fused_softmax_pinv(q_m, k_m, scale=None, iters=4):
    if q_m.ndim == 4:
        B, H, M, D = q_m.shape
        q_m = q_m.reshape(B * H, M, D)
        k_m = k_m.reshape(B * H, M, D)
        reshape_back = True
    elif q_m.ndim == 3:
        BH, M, D = q_m.shape
        B = H = None
        reshape_back = False
    else:
        raise ValueError("q_m and k_m must have shape [BH,M,D] or [B,H,M,D]")

    if q_m.dtype not in (torch.float16, torch.float32):
        raise TypeError("Only float16 and float32 inputs are supported")
    if k_m.dtype != q_m.dtype:
        raise TypeError("q_m and k_m must have the same dtype")
    if not q_m.is_cuda or not k_m.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    if k_m.shape != q_m.shape:
        raise ValueError("q_m and k_m must have identical shapes")

    q_m = q_m.contiguous()
    k_m = k_m.contiguous()
    BH, M, D = q_m.shape
    scale = D ** -0.5 if scale is None else scale
    fast_fp16 = q_m.dtype == torch.float16

    out = torch.empty((BH, M, M), device=q_m.device, dtype=q_m.dtype)
    bm, bd = triton.next_power_of_2(M), triton.next_power_of_2(D)
    if bd > 128:
        raise ValueError("D must be <= 128")

    fused_softmax_pinv_kernel[(BH,)](
        q_m, k_m, out,
        *q_m.stride(), *k_m.stride(), *out.stride(),
        M=M, D=D, BM=bm, BD=bd, ITERS=iters, SCALE=scale,
        FAST_FP16=fast_fp16,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out.reshape(B, H, M, M) if reshape_back else out