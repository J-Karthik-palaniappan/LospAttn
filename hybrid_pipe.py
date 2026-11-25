import types
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func
import sys
from linear_attention import ATTNS


# ---------------------------------------------------------
# CSV threshold calculation
# ---------------------------------------------------------
def hybrid_thresholds(
    csv_path,
    error_threshold=0.3463,
    rank_threshold=32,
    sparse_threshold=0.2,
):
    df = pd.read_csv(csv_path)

    rank_cols = sorted(
        [
            c for c in df.columns
            if c not in ["timestep", "layer", "head", "window"]
        ],
        key=int,
    )

    ranks = [int(c) for c in rank_cols]

    agg_cols = rank_cols + ["window"]

    df = df.groupby(
        ["timestep", "layer", "head"],
        as_index=False,
    )[agg_cols].mean()

    df["erank"] = [
        next(
            (
                rank
                for rank, error in zip(ranks, row)
                if error < error_threshold
            ),
            512,
        )
        for row in df[rank_cols].to_numpy()
    ]

    t_min = df["timestep"].min()
    t_max = df["timestep"].max()

    if t_min == t_max:
        raise ValueError("Need at least two distinct timesteps.")

    df["t"] = (
        (df["timestep"] - t_min) / (t_max - t_min)
    )

    df = df.sort_values("t", ascending=False)

    lowrank = {}
    sparse = {}

    for (layer, head), group in df.groupby(["layer", "head"]):

        crossed = group["erank"].to_numpy() > rank_threshold

        L = (
            group["t"].iloc[np.flatnonzero(crossed)[0]]
            if crossed.any()
            else 0.0
        )

        crossed = (
            group["window"].to_numpy() < sparse_threshold
        )

        S = (
            group["t"].iloc[np.flatnonzero(crossed)[0]]
            if crossed.any()
            else 0.0
        )

        lowrank[layer, head] = float(L)
        sparse[layer, head] = float(S)

    layers = sorted(df["layer"].unique())
    heads = sorted(df["head"].unique())

    L = np.array([
        [lowrank[l, h] for h in heads]
        for l in layers
    ])

    S = np.array([
        [sparse[l, h] for h in heads]
        for l in layers
    ])

    lowrank_pct = np.mean(1.0 - L) * 100
    sparse_pct = np.mean(np.minimum(S, L)) * 100
    softmax_pct = np.mean(np.maximum(L - S, 0.0)) * 100

    print(f"Lowrank: {lowrank_pct:.2f}%")
    print(f"Sparse:  {sparse_pct:.2f}%")
    print(f"Softmax: {softmax_pct:.2f}%")

    S_effective = np.where(
        S > L,
        np.nextafter(L, np.inf),
        S,
    )

    return L.tolist(), S_effective.tolist(), float(t_min), float(t_max),


# ---------------------------------------------------------
# FlashAttention local window
# ---------------------------------------------------------
def flash_window(q, k, v, scale, window_size=256):
    return flash_attn_func(
        q,
        k,
        v,
        window_size=(window_size, window_size),
        causal=False,
        softmax_scale=scale,
        dropout_p=0.0,
    )


def flash_full(q, k, v, scale):
    return flash_attn_func(
        q,
        k,
        v,
        causal=False,
        softmax_scale=scale,
        dropout_p=0.0,
    )


class DiffusersHybridAttentionProcessor(nn.Module):

    def __init__(
        self,
        lowrank_mask,
        sparse_mask,
        myblock,
        attn_type="symmetric_nystrom_mid",
        attn_args=None,
        window_size=256,
        timestep_min=0.0,
        timestep_max=999.0,
    ):
        super().__init__()

        if attn_args is None:
            attn_args = {"landmarks": 32, "iters": 4}

        self.register_buffer(
            "lowrank_thresholds",
            torch.tensor(lowrank_mask, dtype=torch.float32),
        )

        self.register_buffer(
            "sparse_thresholds",
            torch.tensor(sparse_mask, dtype=torch.float32),
        )

        self.lowrank_attn = ATTNS[attn_type](**attn_args)

        self.window_size = window_size
        self.timestep_min = float(timestep_min)
        self.timestep_max = float(timestep_max)
        self.hybrid_timestep = None

        # Routing cache: (timestep, device) -> (lowrank_idx, sparse_idx, soft_idx),
        self._routes = {}
        self.myblock = myblock

    def _route(self, timestep, device):
        key = (timestep, device)
        route = self._routes.get(key)

        if route is None:
            t = torch.tensor(timestep, dtype=torch.float32)
            t = ((t - self.timestep_min) / (self.timestep_max - self.timestep_min)).clamp(0.0, 1.0)

            lowrank_thresholds = self.lowrank_thresholds.cpu()
            sparse_thresholds = self.sparse_thresholds.cpu()

            # Low-rank > sparse > softmax.
            lowrank_mask = t > lowrank_thresholds
            sparse_mask = t < sparse_thresholds

            lowrank_idx = lowrank_mask.nonzero().flatten()
            sparse_idx = (sparse_mask & ~lowrank_mask).nonzero().flatten()
            soft_idx = (~sparse_mask & ~lowrank_mask).nonzero().flatten()

            route = tuple(idx.to(device) for idx in (lowrank_idx, sparse_idx, soft_idx))
            self._routes[key] = route

        return route

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):

        if encoder_hidden_states is not None:
            raise ValueError("Self-attention only.")

        residual = hidden_states

        B, N, C = hidden_states.shape
        H = attn.heads
        D = C // H

        if self.hybrid_timestep is None:
            raise RuntimeError("Missing hybrid timestep.")

        # Cached routing: no GPU syncs after the first call per timestep.
        lowrank_idx, sparse_idx, soft_idx = self._route(
            self.hybrid_timestep, hidden_states.device
        )

        # QKV projections
        x = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            x = attn.spatial_norm(x, temb)
        if getattr(attn, "group_norm", None) is not None:
            x = attn.group_norm(x.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(x)
        k = attn.to_k(x)
        v = attn.to_v(x)

        # Preserve original Diffusers Q/K normalization.
        if getattr(attn, "norm_q", None) is not None:
            q = attn.norm_q(q)

        if getattr(attn, "norm_k", None) is not None:
            k = attn.norm_k(k)

        # Keep FlashAttention's native [B, N, H, D] layout (as in FastAttnProcessor).
        q = q.view(B, N, H, D)
        k = k.view(B, N, H, D)
        v = v.view(B, N, H, D)

        out = torch.empty_like(q)
        scale = attn.scale

        # Low-rank attention (ATTNS kernels expect [B, H, N, D])
        if lowrank_idx.numel() > 0:
            # print("lowrank_idx",self.myblock)
            q_lr = q.index_select(2, lowrank_idx).transpose(1, 2).contiguous()
            k_lr = k.index_select(2, lowrank_idx).transpose(1, 2).contiguous()
            v_lr = v.index_select(2, lowrank_idx).transpose(1, 2).contiguous()

            out[:, :, lowrank_idx] = self.lowrank_attn(
                q_lr,
                k_lr,
                v_lr,
            ).transpose(1, 2).to(out.dtype)

        # Standard softmax attention (flash_attn, no window)
        if soft_idx.numel() > 0:
            # print("softmaax",self.myblock)
            out[:, :, soft_idx] = flash_full(
                q.index_select(2, soft_idx),
                k.index_select(2, soft_idx),
                v.index_select(2, soft_idx),
                scale=scale,
            )

        # Sparse window attention (flash_attn, sliding window)
        if sparse_idx.numel() > 0:
            #print("sparse",self.myblock)
            out[:, :, sparse_idx] = flash_window(
                q.index_select(2, sparse_idx),
                k.index_select(2, sparse_idx),
                v.index_select(2, sparse_idx),
                scale=scale,
                window_size=self.window_size,
            )

        out = out.reshape(B, N, C)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if attn.residual_connection:
            out = out + residual

        out = out / attn.rescale_output_factor

        return out


# ---------------------------------------------------------
# Install hybrid attention in Diffusers DiT
# ---------------------------------------------------------
def build_dit_fast_hybrid(
    pipe,
    csv_path,
    rank_threshold=32,
    error_threshold=0.3463,
    sparse_threshold=0.2,
    csv_type="mse",
    attn_type="symmetric_nystrom_mid",
    attn_args=None,
    window_size=256,
):

    transformer = pipe.transformer

    if not hasattr(transformer, "transformer_blocks"):
        raise TypeError(
            "Expected a Diffusers DiTTransformer2DModel "
            "with transformer_blocks."
        )

    lowrank_masks, sparse_masks, tmin, tmax = hybrid_thresholds(
        csv_path=csv_path,
        error_threshold=error_threshold,
        rank_threshold=rank_threshold,
        sparse_threshold=sparse_threshold,
    )

    blocks = transformer.transformer_blocks

    if len(lowrank_masks) != len(blocks):
        raise ValueError(
            f"CSV contains {len(lowrank_masks)} layers, "
            f"but Diffusers DiT has {len(blocks)} blocks."
        )

    processors = []

    for i, block in enumerate(blocks):

        attn = block.attn1

        if getattr(attn, "fused_projections", False):
            raise RuntimeError(
                "Fused QKV projections are unsupported. "
                "Use unfused projections."
            )

        # if attn.to_k is not attn.to_v:
        #     raise RuntimeError("Unexpected attention projections.")

        if attn.heads != len(lowrank_masks[i]):
            raise ValueError(
                f"Head mismatch at layer {i}: "
                f"model={attn.heads}, CSV={len(lowrank_masks[i])}"
            )

        processor = DiffusersHybridAttentionProcessor(
            lowrank_mask=lowrank_masks[i],
            sparse_mask=sparse_masks[i],
            attn_type=attn_type,
            attn_args=attn_args,
            window_size=window_size,
            timestep_min=tmin,
            timestep_max=tmax,
            myblock=i
        )

        # Move processor buffers and low-rank module to model device/dtype.
        processor = processor.to(
            device=next(transformer.parameters()).device,
            dtype=next(transformer.parameters()).dtype,
        )

        attn.set_processor(processor)

        processors.append(processor)

    # Inject the actual scheduler timestep into every attention processor.
    original_forward = transformer.forward

    # Propagate timestep to all hybrid attention processors.
    def timestep_hook(module, args, kwargs):

        timestep = kwargs.get("timestep", None)

        if timestep is None and len(args) > 1:
            timestep = args[1]

        if timestep is None:
            raise RuntimeError("Missing transformer timestep.")

        # One host read per transformer forward (instead of per layer).
        # All batch entries (incl. CFG halves) share the same timestep.
        if torch.is_tensor(timestep):
            timestep = timestep.reshape(-1)[0].item()

        for processor in processors:
            processor.hybrid_timestep = timestep

        return args, kwargs
    
    hook = transformer.register_forward_pre_hook(
        timestep_hook,
        with_kwargs=True,
    )

    transformer._hybrid_timestep_hook = hook
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    return pipe, processors