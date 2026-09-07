"""Scaling-law fits: IsoFLOP parabolas, power laws, and a joint Chinchilla-style fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .data import D_OUT
from .train import BATCH

FLOPS_PER_ND = 6.0 * D_OUT  # C = FLOPS_PER_ND * n * D   (n = embedding dim, D = tokens)

# The same relation in the units the student works in: C = 6ND with N the PARAMETER count
# (N = D_OUT * embedding dim).  Any two of (C, N, D) fix the third, which is what lets a sweep
# be given as any pair of them.

def compute_of(params, d) -> float:
    """C = 6ND."""
    return 6.0 * float(params) * float(d)


def d_of(c: float, params) -> float:
    """Tokens at compute `c` for a model of `params` parameters."""
    return float(c) / (6.0 * float(params))


def params_of(c: float, d) -> float:
    """Parameters at compute `c` for a run of `d` tokens."""
    return float(c) / (6.0 * float(d))


def steps_for(c: float, n: int) -> int:
    """Steps that make an n-dim model cost exactly `c` flops."""
    return max(1, int(round(c / (FLOPS_PER_ND * n * BATCH))))


def tokens_for(c: float, n: int) -> float:
    return c / (FLOPS_PER_ND * n)


def d_for(c: float, n: int) -> int:
    """Tokens that make an n-dim model cost about `c` flops, as a whole number of batches.

    ``tokens_for`` is the exact real number; this is the one you can actually train on, since
    a run is a whole number of steps of ``BATCH`` tokens.
    """
    return steps_for(c, n) * BATCH


# --------------------------------------------------------------------------- #
# IsoFLOP profile -> (n*, L*)
# --------------------------------------------------------------------------- #
@dataclass
class IsoFit:
    c: float
    n_star: float
    loss_star: float
    coef: np.ndarray
    used: np.ndarray
    clipped: str = ""  # "low"/"high" if the fitted minimum falls outside the n grid,
    #                    "flat" if the profile is not convex over the sampled range


def isoflop_optimum(ns, losses, k: int = 3) -> IsoFit:
    """Parabola in log n through the k+1 points nearest the empirical minimum."""
    ns = np.asarray(ns, float)
    losses = np.asarray(losses, float)
    order = np.argsort(losses)
    keep = np.sort(order[: min(k + 1, len(ns))])
    x, y = np.log(ns[keep]), losses[keep]
    coef = np.polyfit(x, y, 2)
    if coef[0] <= 0:  # not convex over this range -> fall back to the argmin
        i = int(order[0])
        return IsoFit(np.nan, ns[i], losses[i], coef, keep, "flat")
    xs = -coef[1] / (2 * coef[0])
    # only a genuinely out-of-range vertex is a problem: an argmin sitting on a grid
    # edge is fine as long as the parabola places the minimum inside the sampled range
    edge = "low" if xs < np.log(ns.min()) else "high" if xs > np.log(ns.max()) else ""
    xs = np.clip(xs, np.log(ns.min()), np.log(ns.max()))
    return IsoFit(np.nan, float(np.exp(xs)), float(np.polyval(coef, xs)), coef, keep, edge)


def r2_of(y, pred) -> float:
    """Coefficient of determination of `pred` against `y`, both in their own units.

    Reported next to every fit so two forms fitted to the same points can be compared.
    `powerlaw` returns its own r^2 on LOG y (a power law is a line there); this one is on
    whatever scale it is handed, which for a loss fit is nats.
    """
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0


def powerlaw(x, y):
    """y = a x^b ; returns (a, b, r2)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-30)
    return float(np.exp(la)), float(b), float(r2)


def saturating_powerlaw(c, loss, l_inf0=None):
    """L = L_inf + A C^-alpha.  Returns (L_inf, A, alpha).

    All three parameters come from the data: the floor is not an input.  ``l_inf0`` is one
    extra starting point for the multi-start, for a caller that has an independent estimate
    of the floor; it defaults to half the smallest loss seen, which assumes nothing.
    """
    c = np.asarray(c, float)
    loss = np.asarray(loss, float)
    if l_inf0 is None:
        l_inf0 = 0.5 * float(np.min(loss))

    def resid(p):
        l_inf, la, al = p
        return (l_inf + np.exp(la) * c ** (-al)) - loss

    # A cross-entropy floor cannot be negative, and cannot exceed the best loss seen; the
    # starting points are clamped into that window rather than dropped by the solver for
    # being outside it, which is what used to happen to half of them.
    lo_l, hi_l = 0.0, float(np.min(loss))
    starts = sorted({min(max(v, lo_l), hi_l)
                     for v in (0.0, 1.0, l_inf0, float(np.min(loss)) - 0.05)})
    best, best_cost = None, np.inf
    for l0 in starts:
        for al0 in (0.02, 0.05, 0.1, 0.2):
            try:
                r = least_squares(resid, [l0, np.log(max(loss.max() - l0, 1e-3)) + al0 * np.log(c[0]), al0],
                                  bounds=([lo_l, -50, 1e-3], [max(hi_l, lo_l + 1e-9), 50, 2.0]))
            except Exception:
                continue
            if r.cost < best_cost:
                best, best_cost = r.x, r.cost
    l_inf, la, al = best
    return float(l_inf), float(np.exp(la)), float(al)


# --------------------------------------------------------------------------- #
# joint fit: one law for every run, in one of two functional forms
# --------------------------------------------------------------------------- #
# "scaling" is an alias for "kaplan": that form is the one from Scaling Laws for Neural
# Language Models, and both names get typed.
FORMS = ("chinchilla", "kaplan")
_ALIASES = {"scaling": "kaplan", "hoffmann": "chinchilla", "additive": "chinchilla"}


def _form(name: str) -> str:
    key = _ALIASES.get(str(name).lower(), str(name).lower())
    if key not in FORMS:
        raise ValueError(f"unknown form {name!r}; one of {FORMS} "
                         f"(or an alias: {sorted(_ALIASES)})")
    return key


@dataclass
class JointFit:
    """One law fitted to every run, in whichever functional form was asked for.

    chinchilla   L = L_inf + A n^-alpha + B d^-beta          [hoffmann2022training]
    kaplan       L = [(A/n)^(alpha/beta) + B/d]^beta         [kaplan2020scaling]

    Both live in the same five slots because the Kaplan form is the additive one with the
    sum taken inside a power: its A and B are that paper's N_c and D_c, and it has no
    floor, so ``l_inf`` stays 0 there.  Expanded, its two terms still go as n^-alpha and
    d^-beta, which is why ``n_exponent`` is one expression for both.
    """

    l_inf: float
    a: float
    alpha: float
    b: float
    beta: float
    rmse: float
    r2: float = float("nan")
    form: str = "chinchilla"
    # How compute relates to the two axes, C = flops_per_nd * n * d, and the search range for
    # the constrained optimum.  The defaults are the embedding-dimension convention (C = 6 *
    # D_OUT * n * D); `joint_fit` replaces them from the data when it is given each run's
    # compute, which is what makes `optimum` right when n counts PARAMETERS (C = 6ND).
    flops_per_nd: float = FLOPS_PER_ND
    n_lo: float = 4.0
    n_hi: float = 4e5

    def predict(self, n, d):
        n = np.asarray(n, float)
        d = np.asarray(d, float)
        if self.form == "kaplan":
            return ((self.a / n) ** (self.alpha / self.beta) + self.b / d) ** self.beta
        return self.l_inf + self.a * n**-self.alpha + self.b * d**-self.beta

    def __str__(self) -> str:
        r2 = "" if not np.isfinite(self.r2) else f"   r2={self.r2:.4f}"
        if self.form == "kaplan":
            return (f"L = [({self.a:.4g}/N)^({self.alpha:.4f}/{self.beta:.4f}) "
                    f"+ {self.b:.4g}/D]^{self.beta:.4f}{r2}")
        return (f"L = {self.l_inf:.4f} + {self.a:.4g} N^-{self.alpha:.4f} "
                f"+ {self.b:.4g} D^-{self.beta:.4f}{r2}")

    @property
    def label(self) -> str:
        """Short tag for a plot legend."""
        return f"{self.form} fit"

    # compute-optimal allocation:  n* ~ C^(beta/(alpha+beta))
    @property
    def n_exponent(self) -> float:
        return self.beta / (self.alpha + self.beta)

    def optimum(self, c: float, flops_per_nd: float | None = None, n_range=None):
        """(n*, D*, predicted loss) at compute budget c, by 1-d search over log n.

        The constraint is C = ``flops_per_nd`` * n * d, taken from the fit unless overridden.
        """
        k = self.flops_per_nd if flops_per_nd is None else float(flops_per_nd)
        lo, hi = (self.n_lo, self.n_hi) if n_range is None else n_range
        ns = np.exp(np.linspace(np.log(lo), np.log(hi), 4000))
        ds = c / (k * ns)
        ls = self.predict(ns, ds)
        i = int(np.argmin(ls))
        return float(ns[i]), float(ds[i]), float(ls[i])


def joint_fit(n, d, loss, fix_l_inf: float | None = None, c=None,
              form: str = "chinchilla") -> JointFit:
    """Fit one law to every run at once, over both axes.

    ``form`` picks the functional form -- ``"chinchilla"`` (the additive
    L_inf + A n^-alpha + B d^-beta) or ``"kaplan"`` / ``"scaling"``
    ([(A/n)^(alpha/beta) + B/d]^beta, which has no floor). The same points fitted twice is
    the honest way to see how much of an extrapolation is the data and how much is the form
    you chose, so both are here and neither is hidden.

    ``c`` is each run's compute, optional. Given it, the fit infers the constant in
    C = k n d -- 6 when n counts parameters, 6*512 when it counts embedding dimensions -- and
    remembers it, so ``JointFit.optimum`` constrains on the right relation instead of assuming
    one. Without it the embedding-dimension convention is assumed, as before.
    """
    form = _form(form)
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    loss = np.asarray(loss, float)
    if form == "kaplan" and fix_l_inf is not None:
        raise ValueError("the kaplan form has no L_inf to fix; use form='chinchilla' "
                         "(or fix_l_inf=None)")

    if form == "kaplan":
        # p = (log N_c, log D_c, alpha_N, alpha_D).  N_c and D_c are enormous -- the loss is
        # about (N_c/n)^alpha_N, so N_c ~ n * L^(1/alpha_N) -- which is why the starts are
        # built from that relation rather than from the data's own scale.
        def unpack(p):
            lnc, ldc, al, be = p
            return 0.0, np.exp(lnc), al, np.exp(ldc), be

        def resid(p):
            _, nc, al, dc, be = unpack(p)
            return ((nc / n) ** (al / be) + dc / d) ** be - loss

        lmid = max(float(np.median(loss)), 1e-3)
        p0s = []
        for al0 in (0.05, 0.1, 0.2, 0.4):
            for be0 in (0.05, 0.1, 0.2, 0.4):
                # split the loss between the two terms, then invert each one for its scale
                p0s.append([np.log(n.mean()) + np.log(0.5 * lmid) / al0,
                            np.log(d.mean()) + np.log(0.5 * lmid) / be0, al0, be0])
        lo = [-50, -50, 1e-3, 1e-3]
        hi = [400, 400, 3.0, 3.0]
    else:
        def unpack(p):
            if fix_l_inf is None:
                l_inf, la, al, lb, be = p
            else:
                l_inf = fix_l_inf
                la, al, lb, be = p
            return l_inf, np.exp(la), al, np.exp(lb), be

        def resid(p):
            l_inf, a, al, b, be = unpack(p)
            return (l_inf + a * n**-al + b * d**-be) - loss

        p0s = []
        l_starts = ([0.0, 0.5 * float(loss.min()), 0.9 * float(loss.min())]
                    if fix_l_inf is None else [0.0])
        for al0 in (0.1, 0.2, 0.4):
            for be0 in (0.1, 0.2, 0.4):
                for l0 in l_starts:
                    base = [np.log(2.0) + al0 * np.log(n.mean()), al0,
                            np.log(2.0) + be0 * np.log(d.mean()), be0]
                    p0s.append(([l0] + base) if fix_l_inf is None else base)
        lo = ([0.0] if fix_l_inf is None else []) + [-40, 1e-3, -40, 1e-3]
        hi = ([float(loss.min())] if fix_l_inf is None else []) + [40, 3.0, 40, 3.0]

    best, best_cost = None, np.inf
    for p0 in p0s:
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if r.cost < best_cost:
            best, best_cost = r.x, r.cost
    if best is None:
        raise RuntimeError(f"the {form} fit did not converge from any starting point")
    l_inf, a, al, b, be = unpack(best)
    res = resid(best)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    extra = {}
    if c is not None:
        cc = np.asarray(c, float)
        # k from the runs themselves (median, so one mislabelled row cannot move it), and a
        # search range that brackets the fitted data by three decades either side
        extra = dict(flops_per_nd=float(np.median(cc / (n * d))),
                     n_lo=float(n.min()) / 1e3, n_hi=float(n.max()) * 1e3)
    return JointFit(l_inf, a, al, b, be, rmse, r2_of(loss, loss + res), form=form, **extra)
