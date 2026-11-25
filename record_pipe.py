
import csv
import torch
import torch.nn as nn
import sys
from linear_attention import approx_attention_error_triton


class DiffusersRecordAttention(nn.Module):

    def __init__(
        self,
        original_processor,
        block_id,
        landmarks,
        iters=6,
        contain_cls=False,
        chunk_size=512,
        window_size=512,
        reduction_rows=4,
    ):
        super().__init__()

        self.original_processor = original_processor
        self.block_id = block_id

        self.landmarks = tuple(landmarks)
        self.iters = iters
        self.contain_cls = contain_cls
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.reduction_rows = reduction_rows

        # GPU accumulation, indexed by timestep.
        self.timestep_sums = {}
        self.timestep_counts = {}

        self.current_timestep = None

    @torch.no_grad()
    def record(self, attn, hidden_states, temb=None, kwargs=None):

        B, N, C = hidden_states.shape
        H = attn.heads

        x = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            x = attn.spatial_norm(x, temb)

        if getattr(attn, "group_norm", None) is not None:
            x = attn.group_norm(x.transpose(1, 2)).transpose(1, 2)

        q = attn.to_q(x)
        k = attn.to_k(x)

        D = q.shape[-1] // H

        if getattr(attn, "norm_q", None) is not None:
            q = attn.norm_q(q)

        if getattr(attn, "norm_k", None) is not None:
            k = attn.norm_k(k)

        q = q.reshape(B, N, H, D)
        k = k.reshape(B, N, H, D)

        image_rotary_emb = (kwargs or {}).get("image_rotary_emb")

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            q, k = apply_rotary_emb(q, k, image_rotary_emb)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        errors, window_errors = approx_attention_error_triton(
            q, k,
            landmark_sizes=self.landmarks,
            iters=self.iters,
            contain_cls=self.contain_cls,
            chunk_size=self.chunk_size,
            window=self.window_size,
            reduction_rows=self.reduction_rows,
        )

        # [B, H, number of landmarks + window]
        values = torch.cat(
            [
                torch.stack([errors[L] for L in self.landmarks], dim=-1),
                window_errors.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Timestep key (assumes one scheduler timestep per forward pass).
        t = self.current_timestep

        # Accumulate batch sum for this timestep.
        batch_sum = values.sum(dim=0)

        if t not in self.timestep_sums:
            self.timestep_sums[t] = batch_sum.clone()
            self.timestep_counts[t] = B
        else:
            self.timestep_sums[t] += batch_sum
            self.timestep_counts[t] += B

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):

        if encoder_hidden_states is None:
            self.record(attn, hidden_states, temb, kwargs)

        return self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


# ============================================================
# Install recorder
# ============================================================

def record_pipe_attention(
    pipe,
    csv_path,
    landmarks=(32, 64, 128),
    iters=6,
    contain_cls=False,
    chunk_size=512,
    window_size=512,
    reduction_rows=4,
):

    transformer = pipe.transformer
    blocks = transformer.transformer_blocks

    recorders = []

    for i, block in enumerate(blocks):

        attn = block.attn1

        if getattr(attn, "fused_projections", False):
            raise RuntimeError("Disable fused QKV projections first.")

        recorder = DiffusersRecordAttention(
            original_processor=attn.processor,
            block_id=i,
            landmarks=landmarks,
            iters=iters,
            contain_cls=contain_cls,
            chunk_size=chunk_size,
            window_size=window_size,
            reduction_rows=reduction_rows,
        )

        attn.set_processor(recorder)
        recorders.append(recorder)

    # Propagate actual scheduler timestep.
    def timestep_hook(module, args, kwargs):

        timestep = kwargs.get("timestep", None)

        if timestep is None and len(args) > 1:
            timestep = args[1]

        if timestep is None:
            raise RuntimeError("Transformer did not receive a timestep.")

        if torch.is_tensor(timestep):
            timestep = int(timestep.reshape(-1)[0].item())

        for recorder in recorders:
            recorder.current_timestep = timestep

        return args, kwargs

    hook = transformer.register_forward_pre_hook(
        timestep_hook,
        with_kwargs=True,
    )

    pipe._attention_recorders = recorders
    pipe._attention_csv_path = csv_path
    pipe._attention_landmarks = landmarks
    pipe._attention_timestep_hook = hook

    transformer.eval()

    print("Installed attention recorder.")
    print("Layers:", len(blocks))
    print("Heads:", blocks[0].attn1.heads)
    print("Landmarks:", landmarks)

    return pipe


# ============================================================
# Save final CSV and restore processors
# ============================================================

def close_pipe_recorder(pipe):

    recorders = pipe._attention_recorders

    pipe._attention_timestep_hook.remove()

    # Restore original processors.
    for block, recorder in zip(
        pipe.transformer.transformer_blocks,
        recorders,
    ):
        block.attn1.set_processor(recorder.original_processor)

    landmarks = pipe._attention_landmarks

    # Collect all timestep keys.
    timesteps = sorted(
        set().union(*(r.timestep_sums.keys() for r in recorders))
    )

    # Transfer accumulated tensors to CPU only once per recorder.
    cpu_data = []

    for recorder in recorders:

        layer_data = {
            t: (
                recorder.timestep_sums[t] / recorder.timestep_counts[t]
            ).detach().cpu()
            for t in timesteps
            if t in recorder.timestep_sums
        }

        cpu_data.append(layer_data)

    # Write CSV once.
    with open(pipe._attention_csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["timestep", "layer", "head"]
            + [str(L) for L in landmarks]
            + ["window"]
        )

        for t in timesteps:

            for layer, layer_data in enumerate(cpu_data):

                if t not in layer_data:
                    continue

                for head, values in enumerate(layer_data[t].tolist()):

                    writer.writerow([
                        t,
                        layer,
                        head,
                        *values,
                    ])

    print("Saved timestep-wise attention errors:", pipe._attention_csv_path)
    print("Timesteps:", len(timesteps))
    print("Attention recording stopped.")
