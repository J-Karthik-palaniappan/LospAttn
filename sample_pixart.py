import argparse
import json
import torch
from evaluate import load_pixart, test_pixart, ROOT, TEST_PROMPTS, CALIB_PROMPTS
from hybrid_pipe import build_dit_fast_hybrid
from record_pipe import record_pipe_attention, close_pipe_recorder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["pixart1k", "pixart2k"])
    parser.add_argument("--cache-dir", default=None) #huggingface hub directory
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--raw", action="store_true", help="enable to sample from original model")

    # Attention replacement parameters
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--error", type=float, default=0.35)
    parser.add_argument("--sparse", type=float, default=0.35)
    parser.add_argument("--window", type=int, default=512)
    # Lowrank - Linear attention
    parser.add_argument(
        "--attn-type", type=str, default="symmetric_nystrom",
        choices=["nystromformer", "nystromformer_torch", "symmetric_nystrom", 
        "symmetric_nystrom_torch", "performer", "wildcat"]
    )
    parser.add_argument(
        "--attn-args", type=json.loads, default={"landmarks": 64, "iters": 8, "contain_cls": 0}
    ) # defaults to symmetric_nystrom; for other linear attentions, refer their functions


    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],)
    parser.add_argument("--record", action="store_true", help="enable attention recording",)
    args = parser.parse_args()

    if args.output is None:
        if args.raw: args.output = f"images/{args.model}_original"
        else: args.output = f"images/{args.model}"

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    pipe = load_pixart(args.model, dtype=dtype, device=args.device, cache_dir=args.cache_dir)

    if args.record:
        # change recording landmarks and labels as needed
        pipe = record_pipe_attention(
            pipe,
            csv_path=ROOT / f"csvs/{args.model}.csv",
            landmarks=(32, 64, 128),
            iters=8,
            contain_cls=False,
            chunk_size=512,
            window_size=args.window*2,
            reduction_rows=4,
        )
        test_pixart(
            pipe, CALIB_PROMPTS, save_path=ROOT / f"images/{args.model}/rankeval",
            batch=1, n_steps=args.steps, seed=args.seed,
        )
        # csv file is saved only after completion of inference
        close_pipe_recorder(pipe)

    else:
        if not args.raw:
            pipe, _ = build_dit_fast_hybrid(
                pipe,
                csv_path=ROOT / f"csvs/{args.model}.csv",
                rank_threshold=args.rank,
                error_threshold=args.error,
                sparse_threshold=args.sparse,
                window_size=args.window,
                attn_type=args.attn_type,
                attn_args=args.attn_args,
            )
        test_pixart(
            pipe, TEST_PROMPTS,
            save_path=ROOT / args.output,
            batch=args.batch,
            n_steps=args.steps,
            seed=args.seed,
            cfg_scale=args.cfg,
            device=args.device
        )


if __name__ == "__main__":
    main()
