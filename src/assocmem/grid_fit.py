"""Fitting the (N, D) scan and comparing it with the envelope calculation.

Three levels of claim, from most to least fitted:

``free``
    ``E = A N^-a + B D^-b`` with all four free.  The question is whether the measured
    exponents land on ``(alpha - 1, 1 - 1/alpha) = (0.2, 0.1667)``.
``theory-exponents``
    the same with ``a, b`` **fixed** at the predicted values and only the two
    prefactors fitted.  If this fits nearly as well as ``free``, the exponents are
    genuinely the predicted ones rather than a coincidence of a flexible fit; the
    fitted prefactors then say what effective capacity and coverage the model achieves.
``envelope``
    no fitting whatsoever: ``E = l * [tail(cap(N)) + tail(D^(1/alpha))]`` with
    ``l = log d - E[H]`` and ``cap = CAP_PER_H * h`` measured independently by the
    capacity experiment.  Uses the exact Zipf tail sum rather than its continuum
    approximation, so the finite-vocabulary curvature is included.

Everything is fitted to the **excess** loss ``L - L_inf``.  Because ``L_inf`` is known
exactly here it is never a free parameter, which is what usually makes these fits
ill-conditioned when the exponents are as small and as similar as 0.2 and 1/6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .data import D_OUT, GAMMA
from .grid import tail_mass

# Capacity of the trained model, contexts per unit of embedding dimension.  Measured in
# results/capacity.json, where the ratio rises from 33.5 at h = 32 and plateaus at 41.3
# by h = 2048.  CAVEAT: that experiment used d = 256 next tokens and this one uses
# d = 512, and capacity depends on d, so this constant is the weakest input to the
# zero-parameter prediction.  `implied_capacity` below reports what the scan itself
# says it should be.
CAP_PER_H = 41.0

def a_th(gamma: float = GAMMA) -> float:
    """Predicted model-size exponent: capacity ~ N and a tail sum give alpha - 1."""
    return gamma - 1.0


def b_th(gamma: float = GAMMA) -> float:
    """Predicted data exponent: only the first D^(1/alpha) contexts are ever seen."""
    return 1.0 - 1.0 / gamma


A_TH, B_TH = a_th(), b_th()  # the alpha = 1.2 values, used by the single-alpha report


def envelope(n, d, cap_per_h: float = CAP_PER_H, l: float = 3.7746, mode: str = "sum",
             gamma: float = GAMMA):
    """The zero-parameter prediction, from the mass of contexts the model cannot know.

    ``mode="min"`` is the faithful reading of the argument: a context is learned only if
    it is *both* within capacity *and* has been seen, so the unlearned set is
    ``i > min(cap(N), D^(1/alpha))``.  ``mode="sum"`` adds the two tails instead, which
    is what the additive Chinchilla form assumes and which double-counts the contexts
    that fail both tests.
    """
    n, d = np.atleast_1d(np.asarray(n, float)), np.atleast_1d(np.asarray(d, float))
    cap = cap_per_h * n / D_OUT
    seen = d ** (1.0 / gamma)
    if mode == "min":
        return l * np.array([tail_mass(k, gamma) for k in np.minimum(cap, seen)])
    return l * (np.array([tail_mass(k, gamma) for k in cap])
                + np.array([tail_mass(k, gamma) for k in seen]))


# --------------------------------------------------------------------------- fits
# Which functional form?  The envelope argument says a context is learned when it is
# *both* within capacity *and* has been seen, so the failure set is
# ``i > min(cap(N), D^(1/alpha))`` and the loss is a **min of two constraints**, i.e.
# ``E = max(A N^-a, B D^-b)`` -- a kink, not a sum.  The additive Chinchilla form
# ``A N^-a + B D^-b`` is a smooth stand-in for it.  Since the real cliffs are soft, the
# truth is in between, so we also fit the power mean
#
#     E = [ (A N^-a)^q + (B D^-b)^q ]^(1/q)
#
# which is additive at q = 1 and the max as q -> infinity, with q fitted.  This matters
# a lot: the three forms return very different exponents from the same data.
FORMS = ("additive", "max", "power_mean")


@dataclass
class Law:
    a_pref: float
    a_exp: float
    b_pref: float
    b_exp: float
    rmse_log: float  # rms of log(pred) - log(measured), i.e. relative error
    n_pts: int
    form: str = "additive"
    q: float = 1.0
    fixed: bool = False

    def predict(self, n, d):
        t1 = self.a_pref * np.asarray(n, float) ** -self.a_exp
        t2 = self.b_pref * np.asarray(d, float) ** -self.b_exp
        if self.form == "additive":
            return t1 + t2
        if self.form == "max":
            return np.maximum(t1, t2)
        return (t1**self.q + t2**self.q) ** (1.0 / self.q)

    def __str__(self) -> str:
        q = f", q={self.q:.2f}" if self.form == "power_mean" else ""
        return (f"{self.a_pref:7.3f} N^-{self.a_exp:.4f} (+) "
                f"{self.b_pref:6.3f} D^-{self.b_exp:.4f}{q}"
                f"   rel.rms {100 * self.rmse_log:5.2f} %")

    # The compute-optimal exponents come from balancing the two terms, so they are the
    # same for every form; only the prefactor of L*(C) depends on q.
    @property
    def n_exponent(self) -> float:
        return self.b_exp / (self.a_exp + self.b_exp)

    @property
    def loss_exponent(self) -> float:
        return self.a_exp * self.b_exp / (self.a_exp + self.b_exp)

    def optimum(self, c: float):
        ns = np.exp(np.linspace(np.log(512 * 4), np.log(512 * 400_000), 8000))
        ds = c / (6.0 * ns)
        e = self.predict(ns, ds)
        i = int(np.argmin(e))
        return float(ns[i]), float(ds[i]), float(e[i])


def fit_law(n, d, excess, fix_exponents: bool = False, form: str = "additive",
            gamma: float = GAMMA) -> Law:
    """Least squares in log space (relative error), which a power law deserves."""
    n, d, y = (np.asarray(v, float) for v in (n, d, excess))
    ly = np.log(y)
    pm = form == "power_mean"

    def unpack(p):
        if fix_exponents:
            la, lb = p[0], p[1]
            a, b = a_th(gamma), b_th(gamma)
        else:
            la, a, lb, b = p[:4]
        q = float(np.exp(p[-1])) if pm else (1.0 if form == "additive" else np.inf)
        return np.exp(la), a, np.exp(lb), b, q

    def model(p):
        A, a, B, b, q = unpack(p)
        t1, t2 = A * n**-a, B * d**-b
        if form == "additive":
            return t1 + t2
        if form == "max":
            return np.maximum(t1, t2)
        return (t1**q + t2**q) ** (1.0 / q)

    def resid(p):
        return np.log(model(p)) - ly

    best, cost = None, np.inf
    for la0 in (0.0, 1.0, 2.0, 3.0):
        for a0 in ((a_th(gamma),) if fix_exponents else (0.1, 0.2, 0.4, 0.8)):
            base = ([la0, np.log(3.0)] if fix_exponents
                    else [la0, a0, np.log(3.0), b_th(gamma)])
            lo = ([-30, -30] if fix_exponents else [-30, 1e-3, -30, 1e-3])
            hi = ([30, 30] if fix_exponents else [30, 2.0, 30, 2.0])
            if pm:
                for q0 in (0.5, 1.0, 2.0):
                    p0 = base + [np.log(q0)]
                    try:
                        r = least_squares(resid, p0, bounds=(lo + [np.log(0.2)],
                                                             hi + [np.log(60.0)]),
                                          max_nfev=40000)
                    except Exception:
                        continue
                    if r.cost < cost:
                        best, cost = r.x, r.cost
                continue
            try:
                r = least_squares(resid, base, bounds=(lo, hi), max_nfev=40000)
            except Exception:
                continue
            if r.cost < cost:
                best, cost = r.x, r.cost
    A, a, B, b, q = unpack(best)
    rmse = float(np.sqrt(np.mean(resid(best) ** 2)))
    return Law(float(A), float(a), float(B), float(b), rmse, len(y), form, float(q),
               fix_exponents)


def corner_exponents(store: dict):
    """The two local exponents, each read off the corner where its constraint binds.

    Model axis: the two smallest widths at the largest D they share.  Data axis: the two
    smallest D at the largest width they share.  These assume no functional form, which
    is what makes them the right thing to compare with alpha across a sweep.
    """
    m, dn, dd = local_slopes(store)
    hs, ss = sorted({k[0] for k in m}), sorted({k[1] for k in m})
    smax = max(x for x in ss if (hs[0], x) in m and (hs[1], x) in m)
    hmax = max(x for x in hs if (x, ss[0]) in m and (x, ss[1]) in m)
    return (dict(a=dn[(hs[0], hs[1], smax)], at=(hs[0], hs[1], 64 * smax)),
            dict(b=dd[(hmax, ss[0], ss[1])], at=(hmax, 64 * ss[0], 64 * ss[1])))


def implied_capacity(law: Law, l: float = 3.7746) -> float:
    """The capacity constant the fitted N-term corresponds to, contexts per h.

    Matches ``A N^-a`` to ``l * tail(cap)`` at the middle of the fitted range, so it is
    directly comparable with the 41 contexts/h measured by the capacity experiment.
    """
    n0 = 512 * 256.0
    want = law.a_pref * n0**-law.a_exp / l  # target tail mass
    k = np.exp(np.linspace(np.log(10), np.log(1e9), 20000))
    m = np.array([tail_mass(x) for x in k])
    return float(np.interp(-want, -m, k)) / (n0 / D_OUT)


# --------------------------------------------------------------------------- data
def cells_of(store: dict, hs=None, steps=None):
    """(n, d, excess, seed, h, steps) arrays, optionally restricted to a sub-grid."""
    rows = []
    for c in store["cells"].values():
        if hs is not None and c["h"] not in hs:
            continue
        if steps is not None and c["steps"] not in steps:
            continue
        rows.append((D_OUT * c["h"], 64 * c["steps"], c["excess_star"], c["seed"],
                     c["h"], c["steps"]))
    if not rows:
        return tuple(np.array([]) for _ in range(6))
    return tuple(np.array(v) for v in zip(*rows))


def lr_surface(store: dict):
    """log lr* = c0 + c1 log h + c2 log D, fitted on whatever cells exist.

    Used to centre the learning-rate grid at the expensive stage-B points, where a
    five-point sweep is not affordable; the three-point bracket around it still has to
    come out interior or the cell is flagged.
    """
    rows = [(c["h"], 64 * c["steps"], c["lr_star"]) for c in store["cells"].values()
            if c["lr_edge"] in ("", "flat")]
    if len(rows) < 6:
        return None
    h, d, lr = (np.array(v, float) for v in zip(*rows))
    m = np.stack([np.ones_like(h), np.log(h), np.log(d)], 1)
    coef, *_ = np.linalg.lstsq(m, np.log(lr), rcond=None)
    pred = m @ coef
    rms = float(np.sqrt(np.mean((pred - np.log(lr)) ** 2)))

    def f(hh, ss):
        return float(np.exp(coef @ [1.0, np.log(hh), np.log(64 * ss)]))

    f.coef, f.rms, f.n = coef, rms, len(rows)
    return f


def bootstrap(store, hs, steps, n_boot: int = 300, fix: bool = False, seed: int = 0):
    """Resample the seed at every cell and refit, for honest CIs on the exponents."""
    n, d, y, z, h, s = cells_of(store, hs, steps)
    if len(y) == 0:
        return None
    keys = {}
    for i in range(len(y)):
        keys.setdefault((h[i], s[i]), []).append(i)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = [rng.choice(v) for v in keys.values()]
        try:
            law = fit_law(n[pick], d[pick], y[pick], fix_exponents=fix)
            out.append((law.a_exp, law.b_exp, law.a_pref, law.b_pref))
        except Exception:
            continue
    return np.array(out)


def isoflop_profiles(store: dict, min_pts: int = 3):
    """Genuine IsoFLOP profiles, read straight off the grid.

    The grid is powers of two in h and powers of four in steps, so cells with the same
    product h*steps have *exactly* the same flop cost: the anti-diagonals are IsoFLOP
    lines with no interpolation needed.  Fitting a parabola in log N along each gives a
    measured N*(C) -- the same construction the student lab does, and an estimate of the
    compute-optimal exponent that does not go through the joint fit at all.
    """
    groups: dict = {}
    for c in store["cells"].values():
        groups.setdefault(c["h"] * c["steps"], {}).setdefault(c["h"], []).append(
            c["excess_star"])
    out = []
    for prod, by_h in sorted(groups.items()):
        if len(by_h) < min_pts:
            continue
        hs = np.array(sorted(by_h), float)
        ex = np.array([np.mean(by_h[h]) for h in sorted(by_h)])
        x = np.log(D_OUT * hs)
        coef = np.polyfit(x, ex, 2)
        if coef[0] <= 0:
            continue
        xs = -coef[1] / (2 * coef[0])
        edge = not (x.min() <= xs <= x.max())
        out.append(dict(c=6.0 * D_OUT * prod * 64, n_star=float(np.exp(xs)),
                        excess_star=float(np.polyval(coef, xs)), n_pts=len(hs),
                        edge=bool(edge), hs=[int(h) for h in hs]))
    return out


def isoflop_detail(store: dict, min_pts: int = 3, n_curve: int = 25,
                   pad_dex: float = 0.10, gamma: float = GAMMA) -> dict:
    """Everything a figure needs from the IsoFLOP construction, fits included.

    Same anti-diagonals and the same parabola in log N as :func:`isoflop_profiles`, but
    it keeps the points, the parabola itself and the three power laws the minima define:

    ``l_of_c``    L* - L_inf against C, exponent -ab/(a+b)
    ``n_of_c``    N* against C, exponent b/(a+b)
    ``frontier``  L* - L_inf against N* -- the envelope of the profiles, whose exponent
                  is just ``-a`` = -(alpha - 1), since the two C exponents share a
                  denominator.  It is the model-size exponent read straight off the
                  minima, with no compute axis anywhere.

    Profiles whose parabola minimum falls outside the widths measured are kept in the
    output but flagged ``edge``, and excluded from the three fits (an edge minimum is a
    bound, not a measurement).
    """
    a_t, b_t = a_th(gamma), b_th(gamma)
    groups: dict = {}
    for c in store["cells"].values():
        groups.setdefault(c["h"] * c["steps"], {}).setdefault(c["h"], []).append(
            c["excess_star"])

    profiles = []
    for prod, by_h in sorted(groups.items()):
        if len(by_h) < min_pts:
            continue
        hs = sorted(by_h)
        n = np.array([D_OUT * h for h in hs], float)
        ex = np.array([float(np.mean(by_h[h])) for h in hs])
        x = np.log(n)
        coef = np.polyfit(x, ex, 2)
        if coef[0] <= 0:  # concave over these points: no interior minimum to report
            continue
        xs = -coef[1] / (2 * coef[0])
        pad = pad_dex * np.log(10.0)
        xc = np.linspace(x.min() - pad, x.max() + pad, n_curve)
        profiles.append(dict(
            c=6.0 * D_OUT * prod * 64, hs=[int(h) for h in hs],
            n=[float(v) for v in n], excess=[float(v) for v in ex],
            n_seeds=[len(by_h[h]) for h in hs], coef=[float(v) for v in coef],
            n_star=float(np.exp(xs)), excess_star=float(np.polyval(coef, xs)),
            edge=bool(not (x.min() <= xs <= x.max())),
            curve=[[float(np.exp(u)), float(np.polyval(coef, u))] for u in xc]))

    fit = [p for p in profiles if not p["edge"]]
    cs = np.array([p["c"] for p in fit])
    ns = np.array([p["n_star"] for p in fit])
    ls = np.array([p["excess_star"] for p in fit])
    from .fit import powerlaw

    def law(x, y, theory):
        # A slice of the sweep, or a run where every minimum is at an edge, leaves
        # nothing to fit; say so rather than raising out of polyfit.
        if len(x) < 2:
            return dict(pref=None, exp=None, r2=None, theory=theory, n=len(x), span=None)
        a, b, r2 = powerlaw(x, y)
        return dict(pref=a, exp=b, r2=r2, theory=theory, n=len(x),
                    span=[float(np.min(x)), float(np.max(x))])

    return dict(
        profiles=profiles, n_fit=len(fit), n_edge=len(profiles) - len(fit),
        laws=dict(l_of_c=law(cs, ls, -(a_t * b_t) / (a_t + b_t)),
                  n_of_c=law(cs, ns, b_t / (a_t + b_t)),
                  frontier=law(ns, ls, -a_t)),
        alpha=gamma, a_th=a_t, b_th=b_t)


# --------------------------------------------------------------------------- report
def grid_means(store: dict) -> dict:
    """(h, steps) -> mean excess over seeds."""
    acc: dict = {}
    for c in store["cells"].values():
        acc.setdefault((c["h"], c["steps"]), []).append(c["excess_star"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def local_slopes(store: dict):
    """Finite-difference exponents, cell to neighbouring cell.

    These are the only exponent estimates that assume nothing at all about the
    functional form, and the asymptotic corners are where they should match the theory:
    the model axis where N binds (small h, large D) and the data axis where D binds
    (large h, small D).
    """
    m = grid_means(store)
    hs, ss = sorted({k[0] for k in m}), sorted({k[1] for k in m})
    dn, dd = {}, {}
    for (h, s) in m:
        for h2 in hs:
            if h2 > h and (h2, s) in m and not any(h < x < h2 for x in hs):
                dn[(h, h2, s)] = -np.log(m[(h2, s)] / m[(h, s)]) / np.log(h2 / h)
        for s2 in ss:
            if s2 > s and (h, s2) in m and not any(s < x < s2 for x in ss):
                dd[(h, s, s2)] = -np.log(m[(h, s2)] / m[(h, s)]) / np.log(s2 / s)
    return m, dn, dd


def l_measured(store: dict, above: float = 1e8) -> float:
    """The measured cost of a context the model does not know, from the tail strata.

    The envelope calculation assumes this is ``l = log d - E[H]``: the model knows
    nothing about the context, so it pays the full entropy of p(y|x) plus log d.  What
    it actually pays is larger, because an unknown context is not merely uninformed --
    the weights that store the frequent contexts actively mispredict it.
    """
    lo = np.asarray(store["meta"]["bin_lo"], float)
    sel = lo >= above
    vals = [np.mean(np.asarray(c["per_bin"], float)[sel]) for c in store["cells"].values()]
    return float(np.mean(vals))


def effective_contexts(excess: float, l: float) -> float:
    """Invert the tail sum: how many contexts does a loss of `excess` correspond to?"""
    k = np.exp(np.linspace(np.log(2), np.log(1e11), 40000))
    m = np.array([tail_mass(x) for x in k])
    return float(np.interp(-excess / l, -m, k))


def implied_capacity(law: Law, l: float) -> float:
    """Contexts per unit h implied by the fitted N-term, comparable with capacity.json."""
    n0 = 512 * 256.0
    return effective_contexts(law.a_pref * n0**-law.a_exp, l) / (n0 / D_OUT)


def report(store: dict, a_hs=(32, 64, 128, 256, 512),
           a_steps=(100, 400, 1600, 6400, 25_600)) -> dict:
    """Fit the cheap corner, compare with theory, and score the extrapolations."""
    l_inf, l = store["meta"]["l_inf"], store["meta"]["l"]
    n, d, y, z, h, s = cells_of(store, a_hs, a_steps)
    if len(y) == 0:
        print("no stage-A cells yet")
        return {}
    m, dn, dd = local_slopes(store)
    l_hat = l_measured(store)

    print(f"L_inf = {l_inf:.4f} nats, known exactly and never fitted.")
    print(f"fit region: h <= {max(a_hs)}, steps <= {max(a_steps)}  ->  {len(y)} runs, "
          f"C = {6 * n.min() * d.min():.1e} .. {6 * n.max() * d.max():.1e} flops"
          f"   ({len(m)} cells total incl. held out)")

    # ---- 1. local exponents, no functional form assumed --------------------
    print("\n--- 1. local exponents (finite differences, no fitted form) ---")
    hs_all, ss_all = sorted({k[0] for k in m}), sorted({k[1] for k in m})
    smax = max(x for x in ss_all if (hs_all[0], x) in m and (hs_all[1], x) in m)
    ka = (hs_all[0], hs_all[1], smax)
    hmax = max(x for x in hs_all if (x, ss_all[0]) in m and (x, ss_all[1]) in m)
    kb = (hmax, ss_all[0], ss_all[1])
    print(f"  model axis, where N binds (h {ka[0]}->{ka[1]}, D={64 * ka[2]:.2e}): "
          f"a = {dn[ka]:.4f}   vs  alpha-1   = {A_TH:.4f}")
    print(f"  data  axis, where D binds (h={kb[0]}, D {64 * kb[1]:.2e}->{64 * kb[2]:.2e}): "
          f"b = {dd[kb]:.4f}   vs  1-1/alpha = {B_TH:.4f}")
    print(f"  (both are corner values: away from the corners each axis is masked by the "
          f"other bottleneck,\n   so the local slope is smaller -- see the tables in "
          f"the figure)")

    # ---- 2. functional form -----------------------------------------------
    print("\n--- 2. which functional form? fitted on the cheap corner ---")
    laws = {}
    for f in FORMS:
        laws[f] = fit_law(n, d, y, form=f)
        print(f"  {f:<11s} {laws[f]}")
    print(f"  {'theory':<11s} {'':7s} a=-{A_TH:.4f}      {'':6s} b=-{B_TH:.4f}"
          f"      (min of the two constraints, i.e. q -> inf)")
    best = laws["power_mean"]
    print(f"\n  The additive (Chinchilla) form fits acceptably but returns exponents "
          f"{laws['additive'].a_exp / A_TH:.2f}x and\n  "
          f"{laws['additive'].b_exp / B_TH:.2f}x the predicted ones; the power mean fits "
          f"{laws['additive'].rmse_log / best.rmse_log:.0f}x better and returns "
          f"{best.a_exp / A_TH:.2f}x and {best.b_exp / B_TH:.2f}x.")

    # ---- 3. compute-optimal ----------------------------------------------
    print("\n--- 3. compute-optimal allocation, C = 6ND ---")
    for f in FORMS:
        print(f"  {f:<11s} N* ~ C^{laws[f].n_exponent:.4f}   "
              f"L*-L_inf ~ C^-{laws[f].loss_exponent:.4f}")
    print(f"  {'theory':<11s} N* ~ C^{B_TH / (A_TH + B_TH):.4f}   "
          f"L*-L_inf ~ C^-{A_TH * B_TH / (A_TH + B_TH):.4f}")
    profs = isoflop_profiles(store)
    if len(profs) >= 3:
        from .fit import powerlaw
        # A profile whose parabola vertex falls outside the widths tried only bounds
        # n*, it does not measure it, so it is quoted but kept out of the power law.
        good = [p_ for p_ in profs if not p_["edge"]] or profs
        cs = [p_["c"] for p_ in good]
        _, pn, r2n = powerlaw(cs, [p_["n_star"] for p_ in good])
        _, pl, r2l = powerlaw(cs, [p_["excess_star"] for p_ in good])
        print(f"  {'IsoFLOP':<11s} N* ~ C^{pn:.4f}   L*-L_inf ~ C^{pl:.4f}   "
              f"(r2 {r2n:.4f} / {r2l:.4f}, {len(good)} profiles"
              f"{f' of {len(profs)}' if len(good) < len(profs) else ''}, "
              f"C = {min(cs):.1e}..{max(cs):.1e})")
        for p_ in profs:
            print(f"      C={p_['c']:.2e}  h*={p_['n_star'] / D_OUT:7.1f}  "
                  f"excess*={p_['excess_star']:.4f}  from h={p_['hs']}"
                  + ("  <- vertex outside range" if p_["edge"] else ""))
    print("  reference solution, measured the same way: N* ~ C^0.4827, "
          "L*-L_inf ~ C^-0.083")

    # ---- 4. the zero-parameter prediction --------------------------------
    print("\n--- 4. the envelope, with nothing fitted ---")
    print(f"  cost of an unknown context: l = log d - L_inf = {l:.3f} nats predicted, "
          f"{l_hat:.3f} measured\n    (the tail strata: an unknown context is worse "
          f"than uninformed, the model mispredicts it)")
    for lab, ll in (("l predicted", l), ("l measured", l_hat)):
        env = envelope(n, d, l=ll)
        mn = envelope(n, d, l=ll, mode="min")
        print(f"  {lab:<12s} sum of tails: bias {100 * np.mean(np.log(env / y)):+5.1f} %"
              f"   min of tails: bias {100 * np.mean(np.log(mn / y)):+5.1f} %")
    print(f"  implied capacity from the fit: "
          f"{implied_capacity(best, l_hat):.1f} contexts/h"
          f"   (capacity.json, at d=256: {CAP_PER_H:.0f})")
    ke = [(k, effective_contexts(v, l_hat)) for k, v in sorted(m.items())]
    print("  effective contexts known, vs the envelope's min(cap, D^(1/alpha)):")
    for (hh, ss), k in ke[::max(1, len(ke) // 6)]:
        pred = min(CAP_PER_H * hh, (64 * ss) ** (1 / GAMMA))
        print(f"      h={hh:<5d} D={64 * ss:.2e}   known {k:9.0f}   "
              f"envelope {pred:9.0f}   ratio {k / pred:5.2f}")

    # ---- 5. extrapolation -------------------------------------------------
    held = sorted(k for k in m if k[0] not in a_hs or k[1] not in a_steps)
    if held:
        print(f"\n--- 5. extrapolation to {len(held)} held-out cells ---")
        print(f"  {'cell':>14s} {'C':>9s} {'meas':>8s}" +
              "".join(f"{f[:9]:>10s}" for f in FORMS) + f"{'envelope':>10s}")
        for hh, ss in held:
            nn, dd_ = D_OUT * hh, 64 * ss
            v = m[(hh, ss)]
            errs = "".join(f"{100 * (float(laws[f].predict(nn, dd_)) / v - 1):+9.1f}%"
                           for f in FORMS)
            env = float(envelope(nn, dd_, l=l_hat, mode="min")[0])
            print(f"  h{hh:<5d}s{ss:<7d} {6 * nn * dd_:9.2e} {v:8.4f}{errs}"
                  f"{100 * (env / v - 1):+9.1f}%")
        nn = np.array([D_OUT * k[0] for k in held], float)
        dd_ = np.array([64 * k[1] for k in held], float)
        vv = np.array([m[k] for k in held])
        print("  rms relative error:      " +
              "".join(f"{100 * np.sqrt(np.mean((np.log(laws[f].predict(nn, dd_) / vv)) ** 2)):9.1f}%"
                      for f in FORMS))
    edges = [k for k, c in store["cells"].items() if c["lr_edge"] not in ("", "flat")]
    if edges:
        print(f"\n  WARNING: {len(edges)} cells never bracketed their lr optimum: "
              f"{', '.join(edges[:6])}{' ...' if len(edges) > 6 else ''}")
    return dict(laws=laws, profiles=profs, l_measured=l_hat, means=m)
