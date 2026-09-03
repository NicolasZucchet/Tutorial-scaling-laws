"""Cached problem instances: one long training stream + fixed eval sets.

The task says p(y|x) is *fixed* across problem instances while embeddings are
*resampled*, so the label stream and the eval conditionals can be generated once
and reused; only ``instance_seed`` (which drives the embeddings) changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import data as D
from .train import EvalSet, Stream

# ASSOCMEM_CACHE relocates the ~250 MB of generated data -- needed when the package is
# installed non-editably (there is no project root then) or to keep it on a mounted
# Drive so a Colab runtime restart does not have to rebuild it.
CACHE = Path(os.environ.get("ASSOCMEM_CACHE",
                            Path(__file__).resolve().parents[2] / "results" / "cache"))


def _p(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


# The stream is stored in two pieces that are only ever *appended*, never rewritten.
#
# `sample_tokens` sizes its rejection-sampling chunk from the requested length, so its
# RNG stream desynchronises with `m`: the first 4M tokens of a 26M draw are NOT the
# first 4M tokens of a 4M draw.  Raising a single MASTER_TOKENS and regenerating would
# therefore silently change the problem instance under every result already recorded
# (REPORT.md, the reference hero runs, results/ledger.jsonl) with no error anywhere.
#
# So the original 4M-token file stays byte-for-byte as it is and is the canonical
# prefix; longer streams are the base file concatenated with a separately-stored,
# independently-seeded extension segment.  Nesting then holds by construction rather
# than by luck, because short and long streams share the same bytes.  Both segments
# are i.i.d. draws from p(x) with independent seeds, so the concatenation is a valid
# i.i.d. stream and the labels stay i.i.d. given the tokens.
MASTER_TOKENS = 4_000_000  # canonical prefix -- NEVER regenerate; see above
EXT_TOKENS = 22_214_400  # appended segment, stored separately
TOTAL_TOKENS = MASTER_TOKENS + EXT_TOKENS  # 26_214_400 = 4^7 * 1600 tokens = the top D rung

# Fingerprint of the canonical prefix, asserted by tests/test_stream.py.  If this ever
# stops matching, the instance has been regenerated and every recorded loss is stale.
BASE_FINGERPRINT = dict(hi=(0, 0, 0), lo=(97, 23, 41), y=(150, 486, 464))


def _base_file(seed: int) -> Path:
    return _p(f"stream_master_s{seed}.npz")


def _ext_file(seed: int) -> Path:
    return _p(f"stream_ext_s{seed}.npz")


def build_extension(seed: int = 0, overwrite: bool = False) -> Path:
    """Draw tokens ``MASTER_TOKENS .. TOTAL_TOKENS`` into their own file (~5 min, 265 MB).

    Refuses to clobber an existing extension, and never opens the base file for
    writing -- the two guarantees that keep older results valid.
    """
    f = _ext_file(seed)
    if f.exists():
        if not overwrite:
            return f
        raise FileExistsError(
            f"{f} exists.  Overwriting it invalidates every result that used tokens "
            f"beyond {MASTER_TOKENS:,}; delete it by hand if that is really intended.")
    tok = D.sample_tokens(EXT_TOKENS, seed=3000 + seed)
    y = D.sample_labels(tok, seed=4000 + seed)
    hi, lo = D.split_u32(tok)
    tmp = f.with_name(f.name + ".partial.npz")  # savez appends .npz unless present
    np.savez(tmp, hi=hi, lo=lo, y=y)
    tmp.rename(f)  # atomic: a killed run leaves no half-written extension
    return f


# The alpha sweep needs a stream per tail exponent.  Those files are new, so there is
# no fingerprint to protect and no need for the two-file scheme above -- but the sampler
# is still not prefix-nested in the requested length, so each one is generated ONCE at a
# fixed canonical length (stage A's largest D) and only ever read as a prefix.  Needing
# more tokens at some alpha means adding an extension file, not enlarging this.
ALPHA_TOKENS = 1_638_400


def _alpha_file(seed: int, gamma: float) -> Path:
    return _p(f"stream_g{gamma:.3f}_s{seed}.npz")


def _write_stream(f: Path, n: int, seed: int, gamma: float) -> None:
    tok = D.sample_tokens(n, seed=seed, gamma=gamma)
    y = D.sample_labels(tok, seed=seed + 1000)
    hi, lo = D.split_u32(tok)
    tmp = f.with_name(f.name + ".partial.npz")
    np.savez(tmp, hi=hi, lo=lo, y=y)
    tmp.rename(f)


def _get_alpha_stream(n_tokens: int, seed: int, gamma: float) -> Stream:
    """Same append-only discipline as the alpha = 1.2 stream, for the same reason.

    The base file is generated at ALPHA_TOKENS and never regrown; asking for more
    appends a separately seeded extension, so a longer request stays a superset of
    every shorter one and results measured earlier remain valid.
    """
    if n_tokens > TOTAL_TOKENS:
        raise ValueError(f"the gamma={gamma:.3f} stream tops out at "
                         f"{TOTAL_TOKENS:,} tokens; asked for {n_tokens:,}")
    f = _alpha_file(seed, gamma)
    if not f.exists():
        _write_stream(f, ALPHA_TOKENS, 1000 + seed, gamma)
    z = np.load(f)
    if n_tokens <= ALPHA_TOKENS:
        return Stream(z["hi"][:n_tokens], z["lo"][:n_tokens], z["y"][:n_tokens])
    e = f.with_name(f"stream_g{gamma:.3f}_ext_s{seed}.npz")
    if not e.exists():
        _write_stream(e, TOTAL_TOKENS - ALPHA_TOKENS, 3000 + seed, gamma)
    ze = np.load(e)
    k = n_tokens - ALPHA_TOKENS
    return Stream(np.concatenate([z["hi"], ze["hi"][:k]]),
                  np.concatenate([z["lo"], ze["lo"][:k]]),
                  np.concatenate([z["y"], ze["y"][:k]]))


def get_stream(n_tokens: int, seed: int = 0, gamma: float = D.GAMMA) -> Stream:
    """Prefix of the one canonical training stream, so every run sees the same data.

    Prefixes are nested by construction, which is what makes runs at different step
    counts comparable.  Up to ``MASTER_TOKENS`` this reads the original file and
    nothing else; beyond it, the extension segment is appended.
    """
    if gamma != D.GAMMA:
        return _get_alpha_stream(n_tokens, seed, gamma)
    if n_tokens > TOTAL_TOKENS:
        raise ValueError(f"stream holds {TOTAL_TOKENS:,} tokens; asked for "
                         f"{n_tokens:,}.  Raise problem.EXT_TOKENS and rebuild the "
                         f"*extension* (the base file must not be touched).")
    f = _base_file(seed)
    if not f.exists():
        tok = D.sample_tokens(MASTER_TOKENS, seed=1000 + seed)
        y = D.sample_labels(tok, seed=2000 + seed)
        hi, lo = D.split_u32(tok)
        np.savez(f, hi=hi, lo=lo, y=y)
    z = np.load(f)
    if n_tokens <= MASTER_TOKENS:
        return Stream(z["hi"][:n_tokens], z["lo"][:n_tokens], z["y"][:n_tokens])
    e = np.load(build_extension(seed))
    k = n_tokens - MASTER_TOKENS
    return Stream(np.concatenate([z["hi"], e["hi"][:k]]),
                  np.concatenate([z["lo"], e["lo"][:k]]),
                  np.concatenate([z["y"], e["y"][:k]]))


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
