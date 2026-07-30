import random

import torch


def get_wikitext2_batches(
    tokenizer,
    n_calib: int = 32,
    n_realloss: int = 8,
    seqlen: int = 512,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Loads WikiText-2 and slices it into fixed-length token batches.

    Returns (calib_batches, realloss_batches): disjoint sets of (1, seqlen)
    token-id tensors, the first for Hessian calibration, the second held out
    for real-loss monitoring during pruning.
    """
    from datasets import load_dataset

    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(train["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids

    rng = random.Random(seed)
    n_needed = n_calib + n_realloss
    max_start = input_ids.shape[1] - seqlen - 1
    starts = rng.sample(range(max_start), n_needed)

    batches = [input_ids[:, s : s + seqlen] for s in starts]
    return batches[:n_calib], batches[n_calib : n_calib + n_realloss]
