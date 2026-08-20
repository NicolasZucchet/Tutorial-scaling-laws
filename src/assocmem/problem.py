"""Cached problem instances: one long training stream + fixed eval sets.

The task says p(y|x) is *fixed* across problem instances while embeddings are
*resampled*, so the label stream and the eval conditionals can be generated once
and reused; only ``instance_seed`` (which drives the embeddings) changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data as D
from .train import EvalSet, Stream

CACHE = Path(__file__).resolve().parents[2] / "results" / "cache"


def _p(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


MASTER_TOKENS = 4_000_000


def get_stream(n_tokens: int, seed: int = 0) -> Stream:
    """Prefix of the one canonical training stream, so every run sees the same data.

    Built once (~50 s, 46 MB) and then memory-mapped; prefixes are nested by
    construction, which is what makes runs at different step counts comparable.
    """
    if n_tokens > MASTER_TOKENS:
        raise ValueError(f"stream holds {MASTER_TOKENS:,} tokens; asked for "
                         f"{n_tokens:,}. Raise problem.MASTER_TOKENS and rebuild.")
    f = _p(f"stream_master_s{seed}.npz")
    if not f.exists():
        tok = D.sample_tokens(MASTER_TOKENS, seed=1000 + seed)
        y = D.sample_labels(tok, seed=2000 + seed)
        hi, lo = D.split_u32(tok)
        np.savez(f, hi=hi, lo=lo, y=y)
    z = np.load(f)
    return Stream(z["hi"][:n_tokens], z["lo"][:n_tokens], z["y"][:n_tokens])


def get_evalset(n_tokens: int, seed: int = 0) -> EvalSet:
    """Eval tokens x ~ p(x) with their exact conditionals p(.|x)."""
    f = _p(f"eval_s{seed}_{n_tokens}.npz")
    if not f.exists():
        tok = D.sample_tokens(n_tokens, seed=7000 + seed)
        probs = D.conditional(tok)
        ent = -(probs * np.log(np.maximum(probs, 1e-45))).sum(1)
        hi, lo = D.split_u32(tok)
        np.savez(f, hi=hi, lo=lo, probs=probs, entropy=ent)
    z = np.load(f)
    return EvalSet(z["hi"], z["lo"], z["probs"], z["entropy"])
