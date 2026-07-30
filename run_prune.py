import argparse
import csv
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from realloss_pruning.data import get_wikitext2_batches
from realloss_pruning.pruner import SequentialPruner
from realloss_pruning.real_loss import compute_real_loss


def parse_args():
    p = argparse.ArgumentParser(description="CWS-OBS pruning with ELSA-style real-loss tracking")
    p.add_argument("--model", type=str, default="facebook/opt-125m")
    p.add_argument(
        "--method",
        type=str,
        default="cws",
        choices=["cws", "sparsegpt", "cws_realloss"],
        help="'cws' = full-Hessian cancellation-aware OBS, H from local reconstruction (cws_obs.py); "
        "'sparsegpt' = real SparseGPT: diagonal-only selection, full off-diagonal OBS correction "
        "(sparsegpt_obs.py); "
        "'cws_realloss' = same OBS/CWS closed-form selection+correction, but H is the empirical-"
        "Fisher approximation of the REAL loss's Hessian instead of the reconstruction surrogate "
        "(fisher_hessian.py)",
    )
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Used by all methods (None = no chunking, full d_in per layer). SparseGPT/CWS papers "
        "both default to 128.",
    )
    p.add_argument("--damping", type=float, default=0.01)
    p.add_argument("--n-calib", type=int, default=32)
    p.add_argument("--n-realloss", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--no-adaptive", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="prune_log.csv")
    p.add_argument("--save-model", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(args.device)
    model.eval()

    print("Loading WikiText-2 calibration + real-loss batches ...")
    calib_batches, realloss_batches = get_wikitext2_batches(
        tokenizer,
        n_calib=args.n_calib,
        n_realloss=args.n_realloss,
        seqlen=args.seqlen,
    )

    dense_loss = compute_real_loss(model, realloss_batches, args.device)
    print(f"Dense model real loss (held-out): {dense_loss:.4f} (ppl {math.exp(dense_loss):.2f})")

    pruner = SequentialPruner(
        model,
        sparsity=args.sparsity,
        blocksize=args.blocksize,
        damping=args.damping,
        adaptive=not args.no_adaptive,
        method=args.method,
    )

    print(f"Pruning to {args.sparsity:.0%} sparsity, method={args.method}, blocksize={args.blocksize} ...")
    log = pruner.prune(calib_batches, realloss_batches)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_index", "cumulative_sparsity", "real_loss", "perplexity", "damping_used"])
        for entry in log:
            writer.writerow(
                [
                    entry.block_index,
                    f"{entry.cumulative_sparsity:.4f}",
                    f"{entry.real_loss:.4f}",
                    f"{math.exp(entry.real_loss):.4f}",
                    entry.damping_used,
                ]
            )
    print(f"Per-block real-loss log written to {args.out}")

    final_loss = log[-1].real_loss
    print(
        f"Final real loss: {final_loss:.4f} (ppl {math.exp(final_loss):.2f}) "
        f"vs dense {dense_loss:.4f} (ppl {math.exp(dense_loss):.2f})"
    )

    if args.save_model:
        model.save_pretrained(args.save_model)
        print(f"Pruned model saved to {args.save_model}")


if __name__ == "__main__":
    main()
