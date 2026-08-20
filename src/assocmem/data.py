"""Problem definition: Zipf inputs, fixed random conditional outputs, sphere embeddings.

Everything here is *deterministic given a key*, so the (huge) vocabulary never has
to be materialised: token `i`'s output distribution and embedding are generated on
demand from a counter-based hash of `i`.

Conventions
-----------
* tokens are 1-indexed integers in ``[1, V]``
* ``D_OUT = 512`` output classes
* entropies / losses are in **nats**
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

# Fixed salts.  The conditional p(y|x) is FIXED across problem instances, so its
# salt is a hard-coded constant; embeddings are resampled per instance and take a
# user-supplied instance seed.
_SALT_LOGITS = np.uint64(0xA1B2C3D4E5F60789)
_SALT_ENTROPY = np.uint64(0x1234567890ABCDEF)


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
# p(y | x):  random logits + temperature matched to a target entropy
# --------------------------------------------------------------------------- #
def target_entropy(tokens: np.ndarray) -> np.ndarray:
    """H(x) in nats.  exp(H)/d ~ Beta(1, 31), clamped to exp(H) >= 1."""
    v = _hash_stream(tokens, _SALT_ENTROPY, 1)[:, 0]
    u = 1.0 - v ** (1.0 / BETA_B)  # Beta(1, BETA_B)
    return np.log(np.maximum(D_OUT * u, 1.0))


def _entropy_of(beta: np.ndarray, logits: np.ndarray) -> np.ndarray:
    z = beta[:, None] * logits
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(axis=1, keepdims=True)
    p = e / s
    log_z = (m + np.log(s))[:, 0]
    return log_z - (p * z).sum(axis=1)


def _solve_beta(logits: np.ndarray, h: np.ndarray, iters: int = 60) -> np.ndarray:
    """Bisection in log-beta for entropy(softmax(beta*logits)) == h."""
    lo = np.full(h.shape, -20.0)
    hi = np.full(h.shape, 30.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_high = _entropy_of(np.exp(mid), logits) > h  # entropy decreasing in beta
        lo = np.where(too_high, mid, lo)
        hi = np.where(too_high, hi, mid)
    return np.exp(0.5 * (lo + hi))


def conditional(tokens: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """(T,) tokens -> (T, D_OUT) float32 probabilities p(y|x)."""
    out = np.empty((len(tokens), D_OUT), dtype=np.float32)
    for a in range(0, len(tokens), chunk):
        b = min(a + chunk, len(tokens))
        tok = tokens[a:b]
        logits = _norm_ppf(_hash_stream(tok, _SALT_LOGITS, D_OUT))
        beta = _solve_beta(logits, target_entropy(tok))
        z = beta[:, None] * logits
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        out[a:b] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
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


def sample_tokens(m: int, seed: int, gamma: float = GAMMA, p_cut: float = P_CUT) -> np.ndarray:
    """Draw `m` i.i.d. tokens from p(i) proportional to i^-gamma on [1, V]."""
    v = float(vocab_size(gamma, p_cut))
    rng = np.random.default_rng(seed)
    sup = _accept_ratio(np.array([1.0]), gamma)[0]
    tail = (v + 1.0) ** (1.0 - gamma)
    kept: list[np.ndarray] = []
    got = 0
    while got < m:
        k = int((m - got) * 1.8) + 1024
        u = rng.random(k)
        x = (1.0 - u * (1.0 - tail)) ** (-1.0 / (gamma - 1.0))
        i = np.floor(x)
        acc = rng.random(k) * sup < _accept_ratio(i, gamma)
        i = i[acc]
        kept.append(i)
        got += len(i)
    return np.concatenate(kept)[:m].astype(np.int64)


def sample_labels(tokens: np.ndarray, seed: int, uchunk: int = 8192,
                  ochunk: int = 16384) -> np.ndarray:
    """y ~ p(.|x) for each occurrence (fresh draw per occurrence).

    Grouped by distinct token so each conditional is built once (the Zipf stream is
    extremely repetitive), and chunked so memory stays bounded for long streams.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(tokens, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    starts = np.concatenate([[0], np.cumsum(np.bincount(inv_sorted, minlength=len(uniq)))])
    u_all = rng.random(len(tokens))
    y = np.empty(len(tokens), dtype=np.int32)
    for a in range(0, len(uniq), uchunk):
        b = min(a + uchunk, len(uniq))
        cdf = np.cumsum(conditional(uniq[a:b]).astype(np.float64), axis=1)
        cdf[:, -1] = 1.0
        for c0 in range(starts[a], starts[b], ochunk):
            c1 = min(c0 + ochunk, starts[b])
            rows = inv_sorted[c0:c1] - a
            y[order[c0:c1]] = (cdf[rows] < u_all[c0:c1][:, None]).sum(axis=1)
    return np.minimum(y, D_OUT - 1)


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
