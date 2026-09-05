"""Problem definition: Zipf inputs, fixed random conditional outputs, sphere embeddings.

Everything here is *deterministic given a token or a stream position*, so nothing ever
has to be cached: the (huge) vocabulary is never materialised, and token `i`'s output
distribution and embedding are generated on demand from a counter-based hash of `i`.

Conventions
-----------
* tokens are 1-indexed integers in ``[1, V]``
* ``D_OUT = 512`` output classes
* entropies / losses are in **nats**

How p(y|x) is built
-------------------
Each token gets a target entropy ``h(x)`` (hashed, ``exp(h)/d ~ Beta(1, 31)``) and its
conditional is the *universal* profile of that entropy, with the classes permuted by a
token-keyed bijection.  The profile is ``softmax(beta * g*)`` where ``g*`` is the vector
of expected order statistics of ``d`` standard normals -- i.e. the shape a random
softmax-of-gaussians conditional has, with its across-token fluctuation averaged out.

That factorisation is what makes the whole thing cheap.  Matching the entropy of a
*per-token* random logit vector needs a temperature solved per token (a bisection over a
512-wide softmax, which is 95 % of the cost of generating anything).  Matching the
entropy of a *shared* shape is one-dimensional, so it is tabulated once on a grid of
``NH`` entropies and every token is a lookup.  Sampling a label is then O(1): draw a
rank from the profile's CDF, and map it through the token's permutation.

``L_inf = E_x[H(y|x)] = E_x[h(x)]`` depends only on the entropy law, not on the profile
shape, so this leaves the irreducible loss exactly where it was.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import zeta as _hurwitz_zeta

GAMMA = 1.2  # Zipf exponent of the input distribution
P_CUT = 1e-14  # tail cutoff on p(x)
D_OUT = 512  # number of output classes
BETA_B = 31.0  # exp(H)/d ~ Beta(1, BETA_B)  => E[exp(H)] = d / 32 = 16
NH = 4096  # entropy grid: h is snapped to this many levels in [0, log d]

# Fixed salts.  The conditional p(y|x) is FIXED across problem instances, so these are
# hard-coded constants; embeddings are resampled per instance and take a user-supplied
# instance seed.
_SALT_ENTROPY = np.uint64(0x1234567890ABCDEF)
_SALT_PERM = np.uint64(0x5DEECE66D2A9B1C3)
_SALT_TOKEN = np.uint64(0x6F4A7C15E3779B91)

_PHI = np.uint64(0x9E3779B97F4A7C15)
_ODD = np.uint64(0xD1342543DE82EF95)


# --------------------------------------------------------------------------- #
# vocabulary size
# --------------------------------------------------------------------------- #
def _partial_zeta(s: float, v: float) -> float:
    """sum_{i=1}^{V} i^-s  via  zeta(s) - zeta(s, V+1)."""
    return float(_hurwitz_zeta(s, 1.0) - _hurwitz_zeta(s, v + 1.0))


@functools.lru_cache(maxsize=None)
def vocab_size(gamma: float = GAMMA, p_cut: float = P_CUT) -> int:
    """Largest V such that V^-gamma / Z(V) >= p_cut, solved self-consistently."""
    v = 1e12
    for _ in range(200):
        z = _partial_zeta(gamma, v)
        v_new = (1.0 / (p_cut * z)) ** (1.0 / gamma)
        if abs(np.log(v_new / v)) < 1e-14:
            v = v_new
            break
        v = v_new
    return int(np.floor(v))


@functools.lru_cache(maxsize=None)
def zipf_norm(gamma: float = GAMMA, p_cut: float = P_CUT) -> float:
    return _partial_zeta(gamma, float(vocab_size(gamma, p_cut)))


# --------------------------------------------------------------------------- #
# counter-based hashing (numpy side)
# --------------------------------------------------------------------------- #
def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser (uint64 in -> uint64 out)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _hash_stream(tokens: np.ndarray, salt: np.uint64, width: int) -> np.ndarray:
    """(T,) tokens -> (T, width) uniforms in (0,1), deterministic in (token, salt, col)."""
    tok = tokens.astype(np.uint64)[:, None]
    cols = np.arange(width, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        mixed = _splitmix64(tok * np.uint64(0x9E3779B97F4A7C15) + salt)
        state = _splitmix64(mixed ^ (cols * np.uint64(0xD1342543DE82EF95)))
    return (state >> np.uint64(11)).astype(np.float64) * (2.0**-53) + 2.0**-54


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    return ndtri(u)


# --------------------------------------------------------------------------- #
# p(y | x):  universal entropy-indexed profile + token-keyed class permutation
# --------------------------------------------------------------------------- #
def _entropy_of(beta: np.ndarray, logits: np.ndarray) -> np.ndarray:
    z = beta[:, None] * logits
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(axis=1, keepdims=True)
    p = e / s
    log_z = (m + np.log(s))[:, 0]
    return log_z - (p * z).sum(axis=1)


@functools.lru_cache(maxsize=1)
def _profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``NH`` universal profiles, built once (~0.5 s, ~25 MB).

    Returns ``(h, prof, flat)``:

    * ``h[g]``    the entropy profile ``g`` actually has (== the grid target to 1e-13)
    * ``prof[g]`` (NH, d) float32, descending, sums to 1
    * ``flat``    ``(prof.cumsum(1) + g).ravel()``, which is *globally* increasing --
      so one ``searchsorted`` inverts every token's CDF at once, with no per-token
      grouping and no (T, d) temporary.
    """
    i = np.arange(D_OUT)
    g_star = _norm_ppf((D_OUT - i - 0.375) / (D_OUT + 0.25))  # descending order stats
    h_grid = np.linspace(0.0, np.log(D_OUT), NH)

    logits = np.broadcast_to(g_star, (NH, D_OUT))
    lo = np.full(NH, -20.0)
    hi = np.full(NH, 30.0)
    for _ in range(60):  # bisection in log-beta; entropy is decreasing in beta
        mid = 0.5 * (lo + hi)
        too_high = _entropy_of(np.exp(mid), logits) > h_grid
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)
    beta = np.exp(0.5 * (lo + hi))

    z = beta[:, None] * g_star
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    h = -(p * np.log(np.maximum(p, 1e-45))).sum(axis=1)

    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    flat = (cdf + np.arange(NH, dtype=np.float64)[:, None]).ravel()
    return h, p.astype(np.float32), flat


def ent_index(tokens: np.ndarray) -> np.ndarray:
    """Which of the ``NH`` entropy levels token `x` sits at.

    ``exp(h)/d ~ Beta(1, 31)`` clamped to ``exp(h) >= 1``, as before, then snapped to
    the grid -- a quantisation of at most ``log(d)/2(NH-1)`` = 7.6e-4 nats.
    """
    v = _hash_stream(tokens, _SALT_ENTROPY, 1)[:, 0]
    u = 1.0 - v ** (1.0 / BETA_B)  # Beta(1, BETA_B)
    h = np.log(np.maximum(D_OUT * u, 1.0))
    return np.rint(h / np.log(D_OUT) * (NH - 1)).astype(np.int32)


def target_entropy(tokens: np.ndarray) -> np.ndarray:
    """H(x) in nats -- exactly the entropy of the profile the token is given."""
    return _profiles()[0][ent_index(tokens)]


# 512 = 2^9, so a 6-round Feistel network on 9 bits is an O(1), exactly bijective,
# token-keyed permutation of the classes.  The halves are unbalanced (5 and 4 bits) and
# therefore swap sizes every round, which is why the mask alternates.
_M5 = np.uint64(31)
_M4 = np.uint64(15)


def _round_f(tok: np.ndarray, x: np.ndarray, rnd: int, mask: np.uint64) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = tok * _PHI + x * _ODD + np.uint64(rnd) * np.uint64(0x94D049BB133111EB)
        return _splitmix64(z + _SALT_PERM) & mask


def class_perm(rank: np.ndarray, tokens: np.ndarray, rounds: int = 6) -> np.ndarray:
    """Rank within the sorted profile -> class id.  Bijective on [0, 512) per token.

    ``rank`` and ``tokens`` broadcast against each other, so this serves both the
    per-occurrence label draw (both (T,)) and the full conditional ((T, d) against
    (T, 1)).
    """
    t = np.asarray(tokens).astype(np.uint64)
    j = np.asarray(rank).astype(np.uint64)
    left = (j >> np.uint64(4)) & _M5  # 5 bits
    right = j & _M4  # 4 bits
    for k in range(rounds):
        mask = _M5 if k % 2 == 0 else _M4
        left, right = right, (left ^ _round_f(t, right, k, mask)) & mask
    return ((left << np.uint64(4)) | right).astype(np.int32)


def conditional(tokens: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """(T,) tokens -> (T, D_OUT) float32 probabilities p(y|x)."""
    prof = _profiles()[1]
    ranks = np.arange(D_OUT, dtype=np.int64)
    out = np.empty((len(tokens), D_OUT), dtype=np.float32)
    for a in range(0, len(tokens), chunk):
        b = min(a + chunk, len(tokens))
        tok = tokens[a:b]
        cls = class_perm(np.broadcast_to(ranks, (b - a, D_OUT)), tok[:, None])
        np.put_along_axis(out[a:b], cls.astype(np.int64), prof[ent_index(tok)], axis=1)
    return out


# --------------------------------------------------------------------------- #
# sampling x ~ Zipf  (exact, by rejection against the integrated power law)
# --------------------------------------------------------------------------- #
def _accept_ratio(i: np.ndarray, gamma: float) -> np.ndarray:
    """p_i / q_i up to a constant, where q_i = int_i^{i+1} x^-gamma dx.

    ratio(i) = (gamma-1)/i / -expm1(-(gamma-1) log1p(1/i));  ratio(1) is the sup.
    """
    t = 1.0 / i
    return ((gamma - 1.0) * t) / -np.expm1(-(gamma - 1.0) * np.log1p(t))


def _zipf_from_uniforms(u: np.ndarray, tail: float, sup: float,
                        gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """(u0, u1) -> candidate token and whether rejection sampling accepts it."""
    i = np.floor((1.0 - u[:, 0] * (1.0 - tail)) ** (-1.0 / (gamma - 1.0)))
    return i, u[:, 1] * sup < _accept_ratio(i, gamma)


def _envelope(gamma: float, p_cut: float, max_context: int | None) -> tuple[float, float]:
    v = float(vocab_size(gamma, p_cut) if max_context is None else max_context)
    return (v + 1.0) ** (1.0 - gamma), _accept_ratio(np.array([1.0]), gamma)[0]


def stream_tokens(n: int, gamma: float = GAMMA, p_cut: float = P_CUT,
                  max_context: int | None = None, salt: int = 0) -> np.ndarray:
    """Tokens at stream positions ``[0, n)``, as a pure function of the position.

    Prefixes therefore nest *by construction*: the first m tokens of an n-token stream
    are the first m tokens of an m-token stream, for every m <= n.  That is what makes
    runs at different step counts comparable -- and, because it holds without anything
    being written down, it is why the training stream needs no cache file and no
    append-only bookkeeping.

    Contrast ``sample_tokens``, which draws from a numpy Generator and whose rejection
    chunk is sized from ``m``: its output is *not* nested in the requested length.
    """
    tail, sup = _envelope(gamma, p_cut, max_context)
    out = np.zeros(n, dtype=np.int64)
    todo = np.arange(n, dtype=np.int64)
    attempt = 0
    while len(todo):  # ~64 % accept, so this shrinks by ~3x a pass
        with np.errstate(over="ignore"):
            s = _SALT_TOKEN + np.uint64(salt) * _PHI + np.uint64(attempt) * _ODD
        i, acc = _zipf_from_uniforms(_hash_stream(todo, s, 2), tail, sup, gamma)
        out[todo[acc]] = i[acc]
        todo = todo[~acc]
        attempt += 1
    return out


def sample_tokens(m: int, seed: int, gamma: float = GAMMA, p_cut: float = P_CUT,
                  max_context: int | None = None) -> np.ndarray:
    """Draw `m` i.i.d. tokens from p(i) proportional to i^-gamma on [1, V]."""
    tail, sup = _envelope(gamma, p_cut, max_context)
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    got = 0
    while got < m:
        k = int((m - got) * 1.8) + 1024
        i, acc = _zipf_from_uniforms(rng.random((k, 2)), tail, sup, gamma)
        kept.append(i[acc])
        got += int(acc.sum())
    return np.concatenate(kept)[:m].astype(np.int64)


def labels_from_uniforms(tokens: np.ndarray, u: np.ndarray) -> np.ndarray:
    """y ~ p(.|x), one uniform per occurrence.

    The rank is read straight off ``flat`` -- a single ``searchsorted`` over the
    NH*d-entry table, since offsetting row ``g`` by ``g`` makes the stacked CDFs
    globally increasing.  No per-token grouping, no (T, d) temporary, no ``exp``.
    """
    _, _, flat = _profiles()
    g = ent_index(tokens).astype(np.int64)
    rank = np.searchsorted(flat, u + g) - g * D_OUT
    return class_perm(np.minimum(rank, D_OUT - 1), tokens)


def sample_labels(tokens: np.ndarray, seed: int, chunk: int = 4_194_304) -> np.ndarray:
    """y ~ p(.|x) for each occurrence (fresh draw per occurrence).

    Nested in the length of `tokens`, because a PCG64 stream is: the first m draws of
    ``rng.random(n)`` are ``rng.random(m)``.
    """
    rng = np.random.default_rng(seed)
    y = np.empty(len(tokens), dtype=np.int32)
    for a in range(0, len(tokens), chunk):
        b = min(a + chunk, len(tokens))
        y[a:b] = labels_from_uniforms(tokens[a:b], rng.random(b - a))
    return y


# --------------------------------------------------------------------------- #
# embeddings (jax side): unit-sphere, resampled per problem instance
# --------------------------------------------------------------------------- #
def split_u32(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int64 tokens -> (hi, lo) uint32 halves so jax never needs x64."""
    t = tokens.astype(np.uint64)
    return (t >> np.uint64(32)).astype(np.uint32), (t & np.uint64(0xFFFFFFFF)).astype(np.uint32)


@functools.partial(jax.jit, static_argnames=("n",))
def embed(key, hi, lo, n: int):
    """(B,) token halves -> (B, n) float32 unit vectors."""

    def one(h, l):
        k = jax.random.fold_in(jax.random.fold_in(key, h), l)
        v = jax.random.normal(k, (n,), dtype=jnp.float32)
        return v / jnp.linalg.norm(v)

    return jax.vmap(one)(hi, lo)
