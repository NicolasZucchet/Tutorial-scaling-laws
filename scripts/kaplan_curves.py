"""The two "Scaling laws" slides: Kaplan's Figure 2, regenerated from his own fits.

Kaplan et al. [kaplan2020scaling] never published the runs behind Figure 2, so the
curves here are not traced off the figure: they are *evaluated* from the paper's fitted
learning-curve law, so every line on the slide is a line the paper itself claims.

    L(N, S_min) = (Nc/N)^alpha_N + (Sc/S_min)^alpha_S            eq. (1.6) = (5.6)
    B_crit(L)   = B* / L^(1/alpha_B)                             eq. (1.4) = (5.3)
    C_min       = 6 N B_crit(L) S_min                            sec. 6.1

with the fitted constants of Table 3 (alpha_N, alpha_S, Nc, Sc) and eq. (1.4)
(B*, alpha_B).  One run = one model size N, swept over S_min; the tokens it has
processed at that point are D = B_crit(L) S_min, the *minimum* number of tokens needed
to reach L (eq. 5.1-5.2), and its compute is C = 6ND, the deck's own accounting.

That single parameterisation gives both panels of Figure 2 at once, since the two
x-axes differ only by the factor 6N:

    figures/kaplan-tokens-fig.md    loss against tokens processed D   (Fig. 2, left)
    figures/kaplan-compute-fig.md   loss against compute C = 6ND      (Fig. 2, right)

The right panel also carries the **compute-efficient frontier**: at each loss, the
compute of the *cheapest* model size that reaches it, minimised over a continuum of N
rather than over the six plotted runs.  So the envelope is exact, not hand-drawn, and
the diamond on each run marks where that run is the compute-optimal one -- visibly
short of its own plateau, which is the point the slide makes.

    uv run python scripts/kaplan_curves.py            # fits, ranges, sanity checks
    uv run python scripts/kaplan_curves.py --write     # + the two figures/*.md

Both figures are generated and say so in a header comment: do not edit them by hand.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_TOKENS = ROOT / "figures/kaplan-tokens-fig.md"
FIG_COMPUTE = ROOT / "figures/kaplan-compute-fig.md"

# ---------------------------------------------------------------- the paper's numbers
#
# Kaplan et al. 2020, "Scaling Laws for Neural Language Models", arXiv:2001.08361.
# Table 3 ("Fits to L(N,S)") for the learning-curve law, eq. (1.4) for the critical
# batch size.  Nothing here is fitted by us.
ALPHA_N, ALPHA_S = 0.077, 0.76      # Table 3
NC, SC = 6.5e13, 2.1e3              # Table 3: non-embedding parameters, steps
B_STAR, ALPHA_B = 2e8, 0.21         # eq. (1.4): tokens
# eq. (1.3), quoted only as a cross-check on the envelope we compute:
#   L(C_min) = (C_c^min / C_min)^alpha_C^min,  alpha ~ 0.050, C_c^min ~ 3.1e8 PF-days
CC_MIN_PFD, ALPHA_C_MIN = 3.1e8, 0.050
PF_DAY = 8.64e19                    # FLOPs in one petaflop-day (1e15 x 86400 s)

# The six runs. The paper spans 1e3 to 1e9 non-embedding parameters; six round decades
# fill the deck's six-step ramp, and the 1e3 end is the one the paper's own fits
# exclude (1-layer models, see Fig. 13), so it is the one to drop.
SIZES = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
LABELS = ("10k", "100k", "1M", "10M", "100M", "1B")
# Cheap to expensive, small to large: the deck's single-hue ramp, since N is *ordered*.
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#256abf", "#184f95", "#0d366b")
SURFACE = "#fcfcfb"
FRONTIER = "#c0392b"

# Where to start and stop each run.  eq. (1.6) is a fit to the *power-law* part of a
# learning curve -- the paper says outright that it breaks down very early in training
# -- so the curves start just under the initial loss of a uniform predictor
# (ln 50257 = 10.8 nats) and stop 1 % above their own converged loss.
L_TOP = 10.5
PLATEAU_TOL = 0.01

# ---------------------------------------------------------------- geometry
#
# The figure lives in the wider column of a `columns: 3/2` slide, so it is nearly
# square rather than the 3:1 letterbox the IsoFLOP slide uses.  Text sizes come from
# `.plot-fig` in assets/slides.css (22px body, 20px `.pf-small`) and the viewBox is
# chosen so that one viewBox unit renders as roughly one pixel: at any other scale the
# axis type stops matching the rest of the deck.
# The height is the binding constraint, not the width: the slide's lead question and
# the legend take ~145 px off the top of a 578 px column, so anything taller than a
# 2:1 box renders past the bottom of the frame and `.col` clips it away entirely.
W, H = 680, 414
YT, YB = 14, 330          # plot box, top and bottom
X0, X1 = 96, 664          # plot box, left and right


class Log:
    """A log10 axis mapping data to viewBox units."""

    def __init__(self, lo: float, hi: float, p0: float, p1: float):
        self.lo, self.hi, self.p0, self.p1 = lo, hi, p0, p1

    def __call__(self, v):
        t = (np.log10(v) - np.log10(self.lo)) / (np.log10(self.hi) - np.log10(self.lo))
        return self.p0 + t * (self.p1 - self.p0)


# ---------------------------------------------------------------- the laws


def l_inf(n):
    """The converged loss of a model of size N: the first term of eq. (1.6)."""
    return (NC / n) ** ALPHA_N


def loss(n, s):
    """eq. (1.6): loss of a model of size N after S_min large-batch steps."""
    return l_inf(n) + (SC / s) ** ALPHA_S


def steps_for(n, target):
    """Invert eq. (1.6): the S_min at which model N first reaches `target`."""
    return SC / (target - l_inf(n)) ** (1.0 / ALPHA_S)


def b_crit(l):
    """eq. (1.4): the critical batch size, in tokens, at loss L."""
    return B_STAR / l ** (1.0 / ALPHA_B)


def run(n, points: int = 120):
    """One training run: (tokens processed, compute in FLOPs, loss), swept over S_min.

    Tokens are counted as D = B_crit(L) S_min, the minimum number a run needs to reach
    L (eq. 5.1-5.2 define B_crit as exactly that ratio), and compute as C = 6ND, which
    is the paper's C_min for this model size.
    """
    s = np.geomspace(steps_for(n, L_TOP),
                     steps_for(n, (1 + PLATEAU_TOL) * l_inf(n)), points)
    l = loss(n, s)
    d = b_crit(l) * s
    return d, 6 * n * d, l


def frontier(losses, n_lo: float = 1e2, n_hi: float = 1e15, grid: int = 4000):
    """The compute-efficient frontier: min over N of C_min at each target loss.

    At a fixed loss the run of size N needs S_min = Sc / (L - L_inf(N))^(1/alpha_S)
    steps, hence C = 6 N B_crit(L) S_min; only the N-dependence matters, so this is a
    one-dimensional minimisation done on a log grid in N.  Returns (C in FLOPs, N*).
    """
    ns = np.geomspace(n_lo, n_hi, grid)
    out_c, out_n = [], []
    for l in np.atleast_1d(losses):
        gap = l - l_inf(ns)
        ok = gap > 1e-12
        n, gap = ns[ok], gap[ok]
        c = 6 * n * b_crit(l) * SC / gap ** (1.0 / ALPHA_S)
        i = int(np.argmin(c))
        out_c.append(c[i])
        out_n.append(n[i])
    return np.array(out_c), np.array(out_n)


def frontier_loss(n_star: float, lo: float = 1.9, hi: float = 9.5) -> float:
    """The loss at which a model of size `n_star` is the compute-optimal one.

    A higher target loss is an easier target, reached most cheaply by a *smaller* model,
    so N*(L) decreases monotonically and a plain bisection on the frontier's own argmin
    suffices.
    """
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if frontier(mid)[1][0] < n_star:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- svg pieces
#
# Same vocabulary as figures/isoflop-figure.md: one hairline axis per side, ticks and
# labels outside the plot box, no grid, a rotated y title, dashed for anything derived.


def pow10(v: float) -> str:
    """An axis number as a power of ten: `10` with a raised exponent."""
    e = int(round(np.log10(v)))
    return f'10<tspan dx="1" dy="-9" font-size="0.72em">{e}</tspan>'


def plain(v: float) -> str:
    return f"{v:g}"


def frame(sx: Log, sy: Log, xlab: str, ylab: str, x_ticks, y_ticks,
          x_fmt=pow10, y_fmt=plain) -> list[str]:
    out = [f'<line class="pf-axis" x1="{X0}" y1="{YB}" x2="{X1}" y2="{YB}"/>',
           f'<line class="pf-axis" x1="{X0}" y1="{YB}" x2="{X0}" y2="{YT}"/>']
    for v in x_ticks:
        x = sx(v)
        out += [f'<line class="pf-axis" x1="{x:.0f}" y1="{YB}" x2="{x:.0f}" '
                f'y2="{YB + 8}"/>',
                f'<text class="pf-muted pf-small" x="{x:.0f}" y="{YB + 32}" '
                f'text-anchor="middle">{x_fmt(v)}</text>']
    for v in y_ticks:
        y = sy(v)
        out += [f'<line class="pf-axis" x1="{X0 - 8}" y1="{y:.0f}" x2="{X0}" '
                f'y2="{y:.0f}"/>',
                f'<text class="pf-muted pf-small" x="{X0 - 14}" y="{y + 7:.0f}" '
                f'text-anchor="end">{y_fmt(v)}</text>']
    cy = (YT + YB) / 2
    out += [f'<text class="pf-muted pf-small" x="{(X0 + X1) / 2:.0f}" y="{YB + 62}" '
            f'text-anchor="middle">{xlab}</text>',
            f'<text class="pf-muted pf-small" transform="rotate(-90 {X0 - 72:.0f} '
            f'{cy:.0f})" x="{X0 - 72:.0f}" y="{cy:.0f}" text-anchor="middle">'
            f"{ylab}</text>"]
    return out


def polyline(xs, ys, sx: Log, sy: Log) -> str:
    return "M " + " L ".join(f"{sx(x):.1f} {sy(y):.1f}" for x, y in zip(xs, ys))


def diamond(x: float, y: float, r: float = 5.5) -> str:
    return (f"M {x:.1f} {y - r:.1f} L {x + r:.1f} {y:.1f} L {x:.1f} {y + r:.1f} "
            f"L {x - r:.1f} {y:.1f} Z")


def along(p0, p1, t: float, text: str, off: float, colour_class: str) -> str:
    """A label laid along a line and lifted clear of it, as in the IsoFLOP figure."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    norm = float(np.hypot(dx, dy))
    x, y = x0 + t * dx + off * dy / norm, y0 + t * dy - off * dx / norm
    turn = float(np.degrees(np.arctan2(dy, dx)))
    return (f'<text class="{colour_class} pf-small" x="{x:.1f}" y="{y:.1f}" '
            f'transform="rotate({turn:.1f} {x:.1f} {y:.1f})" text-anchor="middle">'
            f"{text}</text>")


def legend_html() -> str:
    spans = "\n".join(
        f'<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true">'
        f'<line x1="1" y1="5" x2="29" y2="5" stroke="{c}" stroke-width="2.4"/>'
        f"</svg>{lab}</span>"
        for lab, c in zip(LABELS, RAMP))
    return ('<div class="cap-legend kap-legend">\n'
            '<span class="num">parameters <em>N</em></span>\n'
            f"{spans}\n</div>")


# ---------------------------------------------------------------- the two figures

Y_TICKS = (2.5, 3, 4, 6, 8, 10)
Y_LO, Y_HI = 2.2, 11.0

TOKENS_LO, TOKENS_HI = 3e5, 6e12
COMPUTE_LO, COMPUTE_HI = 2e10, 2e22
TOKEN_TICKS = (1e6, 1e8, 1e10, 1e12)
COMPUTE_TICKS = (1e12, 1e15, 1e18, 1e21)

ARIA_TOKENS = ("Test loss against the number of tokens processed, log-log, one "
               "training curve per model size from ten thousand to one billion "
               "parameters. Each curve falls as a power law and then flattens onto "
               "its own converged loss; the larger the model, the further left its "
               "curve sits, so a larger model reaches any given loss after fewer "
               "tokens.")
ARIA_COMPUTE = ("Test loss against training compute, log-log, the same six training "
                "curves. Their lower-left envelope is the compute-efficient "
                "frontier, a straight line in log-log with slope minus 0.052; a "
                "diamond marks the point on each curve at which that model size is "
                "the compute-optimal one, well above the curve's own plateau.")


def header(kind: str) -> str:
    """The generated-file banner: what this draws, and every constant behind it."""
    return "\n".join([
        "<!-- Generated by scripts/kaplan_curves.py --write -- do not edit by hand.",
        "",
        f"     {kind}",
        "",
        "     Kaplan et al. 2020, arXiv:2001.08361, Figure 2, regenerated from the",
        "     paper's own fits rather than traced off the published figure:",
        "",
        "       L(N, S_min) = (Nc/N)^alpha_N + (Sc/S_min)^alpha_S     eq. (1.6)/(5.6)",
        "       B_crit(L)   = B* / L^(1/alpha_B)                      eq. (1.4)/(5.3)",
        "       D           = B_crit(L) S_min                         eq. (5.1)-(5.2)",
        "       C           = 6 N D                                   sec. 2.1, 6.1",
        "",
        f"     alpha_N = {ALPHA_N}, alpha_S = {ALPHA_S}, Nc = {NC:.1e} non-embedding",
        f"     parameters, Sc = {SC:.1e} steps (Table 3, 'Fits to L(N,S)');",
        f"     B* = {B_STAR:.0e} tokens, alpha_B = {ALPHA_B} (eq. 1.4).",
        "",
        "     The frontier is the exact envelope of this family -- C minimised over a",
        "     continuum of N at each loss -- and comes out as L proportional to",
        "     C^-0.052, against the -0.054 the paper predicts in eq. (6.4) and the",
        "     -0.050 it measures in eq. (1.3).",
        "-->",
    ])


def tokens_svg() -> str:
    sx = Log(TOKENS_LO, TOKENS_HI, X0, X1)
    sy = Log(Y_LO, Y_HI, YB, YT)
    s = [f'<svg class="plot-fig" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="{ARIA_TOKENS}">',
         "<defs>",
         f'<clipPath id="kt-box"><rect x="{X0}" y="{YT - 10}" width="{X1 - X0}" '
         f'height="{YB - YT + 10}"/></clipPath>',
         "</defs>",
         *frame(sx, sy, 'tokens processed <tspan class="pf-var">D</tspan>', "loss",
                TOKEN_TICKS, Y_TICKS),
         '<g clip-path="url(#kt-box)">']
    for n, col in zip(SIZES, RAMP):
        d, _, l = run(n)
        s.append(f'<path d="{polyline(d, l, sx, sy)}" fill="none" stroke="{col}" '
                 f'stroke-width="2.6" stroke-linecap="round"/>')
    s += ["</g>", "</svg>"]
    return "\n".join(s)


def compute_svg() -> str:
    sx = Log(COMPUTE_LO, COMPUTE_HI, X0, X1)
    sy = Log(Y_LO, Y_HI, YB, YT)
    s = [f'<svg class="plot-fig" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="{ARIA_COMPUTE}">',
         "<defs>",
         f'<clipPath id="kc-box"><rect x="{X0}" y="{YT - 10}" width="{X1 - X0}" '
         f'height="{YB - YT + 10}"/></clipPath>',
         "</defs>",
         *frame(sx, sy,
                'compute <tspan class="pf-var">C</tspan> = 6<tspan class="pf-var">ND'
                "</tspan> (FLOPs)", "loss", COMPUTE_TICKS, Y_TICKS),
         '<g clip-path="url(#kc-box)">']
    for n, col in zip(SIZES, RAMP):
        _, c, l = run(n)
        s.append(f'<path d="{polyline(c, l, sx, sy)}" fill="none" stroke="{col}" '
                 f'stroke-width="2.6" stroke-linecap="round"/>')
    s.append("</g>")

    # The frontier, and the point on each run where that run is the optimal one.  One
    # reveal group with an explicit index so it lands on the same click as the prose
    # step about compute-optimal training stopping short of convergence.
    ends = [frontier_loss(n) for n in SIZES]
    span = np.geomspace(max(ends) * 1.10, min(ends) / 1.10, 40)
    fc, _ = frontier(span)
    s += ['<g class="fragment" data-fragment-index="1">',
          '<g clip-path="url(#kc-box)">',
          f'<path d="{polyline(fc, span, sx, sy)}" fill="none" stroke="{FRONTIER}" '
          f'stroke-width="2.6" stroke-dasharray="7 5" stroke-linecap="round"/>',
          "</g>"]
    for n, col, l in zip(SIZES, RAMP, ends):
        c = frontier(l)[0][0]
        s.append(f'<path d="{diamond(sx(c), sy(l))}" fill="{col}" stroke="{SURFACE}" '
                 f'stroke-width="1.6"/>')
    p0 = (float(sx(fc[0])), float(sy(span[0])))
    p1 = (float(sx(fc[-1])), float(sy(span[-1])))
    # The label hugs the underside of the frontier near its left end: that wedge --
    # below the frontier, left of every plateau -- is the one empty part of the plot,
    # and above the line the label would have to cross five descending curves.
    s += [along(p0, p1, 0.30, "compute-efficient frontier", -50.0,
                "pf-red pf-strong"),
          "</g>", "</svg>"]
    return "\n".join(s)


def body(svg: str) -> str:
    return legend_html() + "\n\n" + svg


def write() -> None:
    FIG_TOKENS.write_text(
        header("Loss against tokens processed (Kaplan Fig. 2, left panel).")
        + "\n\n" + body(tokens_svg()) + "\n")
    FIG_COMPUTE.write_text(
        header("Loss against compute, with the compute-efficient frontier "
               "revealed\n     on the second beat (Kaplan Fig. 2, right panel).")
        + "\n\n" + body(compute_svg()) + "\n")
    print(f"wrote {FIG_TOKENS.relative_to(ROOT)} and {FIG_COMPUTE.relative_to(ROOT)}")


def report() -> None:
    print("run ranges (loss from %.1f down to 1%% above the converged loss):" % L_TOP)
    for n in SIZES:
        d, c, l = run(n)
        print(f"  N = {n:8.0e}  L_inf = {l_inf(n):5.3f}  "
              f"D {d[0]:8.2e} -> {d[-1]:8.2e} tokens  "
              f"C {c[0]:8.2e} -> {c[-1]:8.2e} FLOPs")
    lo = min(run(n)[0][0] for n in SIZES), max(run(n)[0][-1] for n in SIZES)
    ci = min(run(n)[1][0] for n in SIZES), max(run(n)[1][-1] for n in SIZES)
    print(f"  tokens axis needs {lo[0]:.2e} .. {lo[1]:.2e} "
          f"(drawn {TOKENS_LO:.0e} .. {TOKENS_HI:.0e})")
    print(f"  compute axis needs {ci[0]:.2e} .. {ci[1]:.2e} "
          f"(drawn {COMPUTE_LO:.0e} .. {COMPUTE_HI:.0e})")

    print("\ncompute-efficient frontier, against the paper's eq. (1.3):")
    ls = np.array([6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.2])
    cs, ns = frontier(ls)
    for l, c, n in zip(ls, cs, ns):
        pfd = c / PF_DAY
        quoted = (CC_MIN_PFD / pfd) ** ALPHA_C_MIN
        print(f"  L = {l:4.2f}  C_min = {pfd:9.2e} PF-days  N* = {n:9.2e}  "
              f"eq.(1.3) would say L = {quoted:5.3f}  ({100 * (quoted / l - 1):+5.1f} %)")
    lx = np.log10(cs / PF_DAY)
    print(f"  fitted envelope exponent {np.polyfit(lx, np.log10(ls), 1)[0]:+.4f} "
          f"(eq. 6.4 predicts -0.054, eq. 1.3 measures -0.050)")
    print(f"  fitted N*(C) exponent    {np.polyfit(lx, np.log10(ns), 1)[0]:+.4f} "
          f"(eq. 6.5 predicts +0.71, eq. 6.1 measures +0.73)")

    print("\ncompute-optimal point of each plotted run (the diamonds):")
    for n in SIZES:
        l = frontier_loss(n)
        print(f"  N = {n:8.0e}  optimal at L = {l:5.3f}, "
              f"{100 * (l / l_inf(n) - 1):4.1f} % above its converged loss "
              f"{l_inf(n):5.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write figures/kaplan-tokens-fig.md and "
                         "figures/kaplan-compute-fig.md")
    args = ap.parse_args()
    report()
    if args.write:
        write()


if __name__ == "__main__":
    main()
