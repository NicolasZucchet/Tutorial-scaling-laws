"""Scaling-law fits: IsoFLOP parabolas, power laws, and a joint Chinchilla-style fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .data import D_OUT
from .train import BATCH

FLOPS_PER_ND = 6.0 * D_OUT  # C = FLOPS_PER_ND * n * D   (D = tokens)


def steps_for(c: float, n: int) -> int:
    """Steps that make an n-dim model cost exactly `c` flops."""
    return max(1, int(round(c / (FLOPS_PER_ND * n * BATCH))))


def tokens_for(c: float, n: int) -> float:
    return c / (FLOPS_PER_ND * n)


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


def powerlaw(x, y):
    """y = a x^b ; returns (a, b, r2)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-30)
    return float(np.exp(la)), float(b), float(r2)


def saturating_powerlaw(c, loss, l_inf0=2.3):
    """L = L_inf + A C^-alpha.  Returns (L_inf, A, alpha)."""
    c = np.asarray(c, float)
    loss = np.asarray(loss, float)

    def resid(p):
        l_inf, la, al = p
        return (l_inf + np.exp(la) * c ** (-al)) - loss

    best, best_cost = None, np.inf
    for l0 in (0.0, 1.0, l_inf0, min(loss) - 0.05):
        for al0 in (0.02, 0.05, 0.1, 0.2):
            try:
                r = least_squares(resid, [l0, np.log(max(loss.max() - l0, 1e-3)) + al0 * np.log(c[0]), al0],
                                  bounds=([-1, -50, 1e-3], [min(loss), 50, 2.0]))
            except Exception:
                continue
            if r.cost < best_cost:
                best, best_cost = r.x, r.cost
    l_inf, la, al = best
    return float(l_inf), float(np.exp(la)), float(al)


# --------------------------------------------------------------------------- #
# joint fit  L(n, D) = L_inf + A n^-alpha + B D^-beta
# --------------------------------------------------------------------------- #
@dataclass
class JointFit:
    l_inf: float
    a: float
    alpha: float
    b: float
    beta: float
    rmse: float

    def predict(self, n, d):
        return self.l_inf + self.a * np.asarray(n, float) ** -self.alpha \
            + self.b * np.asarray(d, float) ** -self.beta

    # compute-optimal allocation:  n* ~ C^(beta/(alpha+beta))
    @property
    def n_exponent(self) -> float:
        return self.beta / (self.alpha + self.beta)

    def optimum(self, c: float):
        """(n*, D*, predicted loss) at compute budget c, by 1-d search over log n."""
        ns = np.exp(np.linspace(np.log(4), np.log(4e5), 4000))
        ds = c / (FLOPS_PER_ND * ns)
        ls = self.predict(ns, ds)
        i = int(np.argmin(ls))
        return float(ns[i]), float(ds[i]), float(ls[i])


def joint_fit(n, d, loss, fix_l_inf: float | None = None) -> JointFit:
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    loss = np.asarray(loss, float)

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

    p0s, lo, hi = [], [], []
    for al0 in (0.1, 0.2, 0.4):
        for be0 in (0.1, 0.2, 0.4):
            for l0 in ([0.0, 1.5, 2.3] if fix_l_inf is None else [0.0]):
                base = [np.log(2.0) + al0 * np.log(n.mean()), al0,
                        np.log(2.0) + be0 * np.log(d.mean()), be0]
                p0s.append(([l0] + base) if fix_l_inf is None else base)
    lo = ([-2] if fix_l_inf is None else []) + [-40, 1e-3, -40, 1e-3]
    hi = ([float(loss.min())] if fix_l_inf is None else []) + [40, 3.0, 40, 3.0]

    best, best_cost = None, np.inf
    for p0 in p0s:
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if r.cost < best_cost:
            best, best_cost = r.x, r.cost
    l_inf, a, al, b, be = unpack(best)
    rmse = float(np.sqrt(np.mean(resid(best) ** 2)))
    return JointFit(l_inf, a, al, b, be, rmse)
