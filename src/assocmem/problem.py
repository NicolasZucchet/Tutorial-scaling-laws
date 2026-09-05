"""The problem instance: one canonical training stream + eval sets, generated on demand.

Nothing here is cached on disk.  ``data.stream_tokens`` makes the token at stream
position ``j`` a pure function of ``j``, and the label a pure function of ``(j, token)``,
so a prefix of a long stream *is* the corresponding short stream and the whole 26M-token
stream regenerates in ~6 s.  That removes what used to be the expensive part of this
module: a 315 MB pair of ``.npz`` files, plus the append-only discipline needed to keep
them valid (the sampler was not nested in the requested length, so a longer stream had
to be a separately-seeded *extension* of a base file that could never be regenerated).

Streams are memoised in-process at the longest length asked for, so repeated
``run_round`` calls in one kernel pay for generation once.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import data as D
from .train import EvalSet, Stream

# Kept because grid.py / emergence.py cache their (much smaller) stratified eval sets
# here, and because the env var is how Colab relocates them off the ephemeral disk.
CACHE = Path(os.environ.get("ASSOCMEM_CACHE",
                            Path(__file__).resolve().parents[2] / "results" / "cache"))

TOTAL_TOKENS = 26_214_400  # 4^7 * 1600 = the top D rung; the stream is defined beyond it
MASTER_TOKENS = TOTAL_TOKENS  # back-compat alias: there is no longer a base/extension split

_streams: dict[tuple[int, float], Stream] = {}


def _p(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def get_stream(n_tokens: int, seed: int = 0, gamma: float = D.GAMMA) -> Stream:
    """Prefix of the one canonical training stream, so every run sees the same data.

    Prefixes are nested by construction, which is what makes runs at different step
    counts comparable.
    """
    key = (seed, gamma)
    have = _streams.get(key)
    if have is None or len(have) < n_tokens:
        tok = D.stream_tokens(n_tokens, gamma=gamma, salt=seed)
        y = D.sample_labels(tok, seed=2000 + seed)
        hi, lo = D.split_u32(tok)
        have = _streams[key] = Stream(hi, lo, y)
    if len(have) == n_tokens:
        return have
    return Stream(have.hi[:n_tokens], have.lo[:n_tokens], have.y[:n_tokens])


def get_evalset(n_tokens: int, seed: int = 0) -> EvalSet:
    """Eval tokens x ~ p(x) with their exact conditionals p(.|x).

    Kept for the scripts and the reference runs; the Lab evaluates on the *stratified*
    set in :mod:`assocmem.grid` instead, which is both cheaper and far less noisy.
    """
    tok = D.sample_tokens(n_tokens, seed=7000 + seed)
    probs = D.conditional(tok)
    ent = -(probs * np.log(np.maximum(probs, 1e-45))).sum(1)
    hi, lo = D.split_u32(tok)
    return EvalSet(hi, lo, probs, ent)
