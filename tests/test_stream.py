"""Guards on the training stream and on p(y|x).

The stream is a pure function of the position in it, and the conditional a pure
function of the token, so there is nothing cached to protect and no frozen file to
fingerprint.  What still has to hold is that the guarantees those functions are
supposed to provide really do:

* prefixes nest at *every* length, so runs at different step counts see the same data;
* the token marginal is the Zipf law it claims to be;
* each token's conditional is a genuine distribution whose entropy is its target --
  which is what pins L_inf, since L_inf = E_x[H(y|x)];
* the O(1) label sampler actually draws from that conditional.

    PYTHONPATH=src uv run python tests/test_stream.py     # ~30 s
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assocmem import data as D  # noqa: E402
from assocmem import problem as P  # noqa: E402


def test_prefixes_nested():
    """Runs at different step counts must see the same data, in the same order.

    This used to need a frozen base file plus an append-only extension, because the
    rejection sampler was not nested in the requested length.  Position hashing makes
    it hold by construction, so it now holds *past* the old 4M boundary too.
    """
    long = P.get_stream(5_000_000)
    for m in (4096, 64_000, 3_999_999, 4_000_001, 4_500_000):
        short = P.get_stream(m)
        assert np.array_equal(short.hi, long.hi[:m]), m
        assert np.array_equal(short.lo, long.lo[:m]), m
        assert np.array_equal(short.y, long.y[:m]), m
    print("  prefixes nested, including across 4M         4096 / 64k / 4M-1 / 4M+1 / 4.5M")


def test_stream_is_a_pure_function_of_position():
    """Regenerating from scratch must give the same bytes -- no hidden RNG state."""
    P._streams.clear()
    a = P.get_stream(100_000)
    P._streams.clear()
    b = P.get_stream(300_000)
    assert np.array_equal(a.lo, b.lo[:100_000])
    assert np.array_equal(a.y, b.y[:100_000])
    print("  regeneration is byte-identical               100k from a 300k rebuild")


def test_token_marginal_is_zipf():
    tok = D.stream_tokens(2_000_000)
    z = D.zipf_norm()
    for i in (1, 2, 5, 20):
        want = i ** -D.GAMMA / z
        got = (tok == i).mean()
        assert abs(got / want - 1) < 0.05, (i, got, want)
    assert tok.min() >= 1 and tok.max() <= D.vocab_size()
    print(f"  token marginal matches i^-{D.GAMMA} / Z          i = 1 / 2 / 5 / 20 to 5 %")


def test_class_permutation_is_bijective():
    """Every token must get a permutation of the 512 classes, not a multiset."""
    for tk in (1, 2, 512, 12345, 10**9, 111_099_971_001):
        out = D.class_perm(np.arange(D.D_OUT), np.full(D.D_OUT, tk, dtype=np.int64))
        assert len(np.unique(out)) == D.D_OUT, tk
    print("  class_perm bijective on [0, 512)             6 tokens incl. the largest")


def test_conditional_is_a_distribution_at_its_target_entropy():
    tok = np.unique(D.stream_tokens(200_000))[:8192]
    p = D.conditional(tok)
    assert np.abs(p.sum(1) - 1.0).max() < 1e-5
    assert (p >= 0).all()
    ent = -(p * np.log(np.maximum(p, 1e-45))).sum(1)
    assert np.abs(ent - D.target_entropy(tok)).max() < 1e-4
    # exp(H)/d ~ Beta(1, 31) => E[exp H] = 16, which is what sets the difficulty
    assert abs(np.exp(D.target_entropy(D.stream_tokens(200_000))).mean() / 16.0 - 1) < 0.15
    print("  p(y|x) sums to 1 at its target entropy       8192 distinct tokens")


def test_labels_are_drawn_from_the_conditional():
    """The O(1) sampler must reproduce p(.|x), not merely something supported on it."""
    tok = D.stream_tokens(4_000_000)
    y = D.sample_labels(tok, seed=2000)
    for tk in (1, 2, 3):
        m = tok == tk
        emp = np.bincount(y[m], minlength=D.D_OUT) / m.sum()
        p = D.conditional(np.array([tk]))[0]
        err = np.abs(emp - p).max()
        assert err < 5.0 / np.sqrt(m.sum()), (tk, m.sum(), err)
    print("  labels match p(.|x) within sampling noise    tokens 1 / 2 / 3")


def test_l_inf_is_the_mean_target_entropy():
    """L_inf depends on the entropy law alone, which is the point of the profile design."""
    from assocmem.grid import strat_evalset

    es = strat_evalset(head=1024, per_bin=64, per_decade=4)
    tok = D.stream_tokens(2_000_000)
    assert abs(es.l_inf - D.target_entropy(tok).mean()) < 0.02
    print(f"  L_inf agrees stratified vs stream-sampled    {es.l_inf:.4f} nats")


if __name__ == "__main__":
    print("stream guards")
    for f in (test_prefixes_nested, test_stream_is_a_pure_function_of_position,
              test_token_marginal_is_zipf, test_class_permutation_is_bijective,
              test_conditional_is_a_distribution_at_its_target_entropy,
              test_labels_are_drawn_from_the_conditional,
              test_l_inf_is_the_mean_target_entropy):
        f()
    print("ok")
