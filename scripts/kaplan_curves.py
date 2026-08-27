"""The two "Scaling laws" slides: Kaplan's Figure 2, digitized off the figure itself.

Kaplan et al. [kaplan2020scaling] never published the runs behind Figure 2, and its
fitted L(N, S_min) is a smooth power law that misses what the figure actually shows --
the flat stretch at the initialization loss, the S-shaped drop, the noise, the plateau
each run settles onto.  So nothing here is evaluated from a law.  Every line comes from
`scripts/kaplan_figure2.py`, which reads the pixels of the published image and writes
`results/kaplan_figure2.json`: 15 curves per panel, each `{n, rgb, x, loss}`.  Read that
script's docstring for the calibration and the caveats -- the two that matter for what
is drawn here are that N is read off the figure's own colourbar and carries about
+-10 %, and that inside the dense bundle a curve is partly painted over by its
neighbours, so it is sampled more sparsely there.

    figures/kaplan-tokens-fig.md    loss against tokens processed D   (Fig. 2, left)
    figures/kaplan-compute-fig.md   loss against compute C, PF-days   (Fig. 2, right)

Six of the fifteen sizes are drawn, roughly evenly spaced in log N across the whole
recovered range, on the deck's six-step blue ramp.  The compute panel also carries the
**compute-efficient frontier**, computed as the true lower envelope of the digitized
curves -- the pointwise minimum over all fifteen, not a fitted line -- with a diamond on
each drawn run at the middle of the stretch of compute over which that run *is* the
envelope.  The fitted law appears nowhere, not even as a comparison: the only numbers
quoted from the paper are in the slide captions, as a contrast to what we measure.

    uv run python scripts/kaplan_curves.py            # ranges, envelope fit, checks
    uv run python scripts/kaplan_curves.py --write     # + the two figures/*.md

Both figures are generated and say so in a header comment: do not edit them by hand.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from _steps import Steps

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "results/kaplan_figure2.json"
FIG_TOKENS = ROOT / "figures/kaplan-tokens-fig.md"
FIG_COMPUTE = ROOT / "figures/kaplan-compute-fig.md"

PF_DAY = 8.64e19            # FLOPs in one petaflop-day (1e15 x 86400 s)

# Which of the fifteen recovered sizes to draw.  Indices into the JSON's curve list,
# which is sorted by N: these six are as evenly spaced in log N as the recovered set
# allows (gaps of 1.3, 1.2, 1.2, 1.2, 0.9 decades) and span it end to end.  The very
# smallest recovered curve (N ~ 5.6e2) is skipped: it is shadowed by its neighbours for
# most of its length, so it only appears below L = 7.9.
PICK = (1, 4, 6, 9, 12, 14)
# Cheap to expensive, small to large: the deck's single-hue ramp, since N is *ordered*.
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#256abf", "#184f95", "#0d366b")
SURFACE = "#fcfcfb"
FRONTIER = "#c0392b"

SMOOTH = 5                  # moving mean, in samples, over the digitized loss
RESAMPLE = 150              # points per drawn path, even in log x
GRID = 1200                 # envelope grid, even in log C

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


# ---------------------------------------------------------------- the digitized data


def load() -> dict:
    if not DATA.exists():
        raise SystemExit(f"{DATA.relative_to(ROOT)} is missing: run "
                         "`uv run python scripts/kaplan_figure2.py` first")
    return json.loads(DATA.read_text())


def clean(curve: dict) -> tuple[np.ndarray, np.ndarray]:
    """One digitized curve as (x, loss), sorted in x and de-jittered.

    The only processing is a `SMOOTH`-sample moving mean of the loss, which takes out
    the single-pixel jitter of the extraction (one pixel is 0.026 nats).  No monotone
    projection, no fit: the flat top, the S-shaped drop and the plateau are the figure's
    own, and so is what noise survives the smoothing.
    """
    x = np.asarray(curve["x"], float)
    y = np.asarray(curve["loss"], float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if SMOOTH > 1 and len(y) > SMOOTH:
        pad = SMOOTH // 2
        y = np.convolve(np.pad(y, pad, mode="edge"), np.ones(SMOOTH) / SMOOTH, "valid")
    return x, y


def resample(x: np.ndarray, y: np.ndarray, n: int = RESAMPLE):
    """The curve on `n` points even in log x: an SVG path needs no more than that."""
    lx = np.linspace(np.log10(x[0]), np.log10(x[-1]), n)
    return 10.0 ** lx, np.interp(lx, np.log10(x), y)


def curves(data: dict, panel: str) -> list[dict]:
    return data["curves"][panel]


def label_of(n: float) -> str:
    """A model size as a rounded label: 1.3k, 26k, 470k, 8.2M, 150M, 1.3B."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.0f}{suf}" if v >= 10 else f"{v:.1f}{suf}"
    return f"{n:.0f}"


# ---------------------------------------------------------------- the envelope


def envelope(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The compute-efficient frontier: the pointwise lower envelope of the 15 curves.

    On a log grid in compute, every recovered curve is interpolated (NaN outside its own
    range) and the minimum taken.  That is all: no law, no fit, no smoothing beyond the
    per-curve de-jitter.  Returns (compute in PF-days, envelope loss, index of the curve
    that owns each grid point).
    """
    cs = curves(data, "compute")
    lo = min(min(c["x"]) for c in cs)
    hi = max(max(c["x"]) for c in cs)
    grid = np.logspace(np.log10(lo), np.log10(hi), GRID)
    rows = []
    for c in cs:
        x, y = clean(c)
        rows.append(np.interp(np.log10(grid), np.log10(x), y,
                              left=np.nan, right=np.nan))
    m = np.vstack(rows)
    with np.errstate(invalid="ignore"):
        env = np.nanmin(m, axis=0)
    owner = np.array([int(np.nanargmin(col)) if not np.all(np.isnan(col)) else -1
                      for col in m.T])
    return grid, env, owner


def plateau(curve: dict) -> float:
    """The converged loss of a run: the lowest loss it reaches in the figure."""
    return float(clean(curve)[1].min())


def owned(grid, env, owner, i: int, l_max: float = 9.5):
    """The stretch of the frontier that curve `i` owns, as a boolean mask.

    Restricted to the part of the envelope that has left the initialization loss: above
    L ~ 9.5 every run is still flat at ln(50257) = 10.8 and "the cheapest model" is
    meaningless.
    """
    return (owner == i) & (env < l_max) & ~np.isnan(env)


def optimum(grid, env, owner, i: int) -> tuple[float, float]:
    """Where model `i` is the compute-optimal one: (compute in PF-days, loss).

    A size owns a *stretch* of the frontier, not a point, so we take the middle of that
    stretch in log compute.  The endpoints are the two switch-overs to the neighbouring
    sizes and are the noisiest part of the extraction, which is the other reason not to
    quote them.
    """
    m = owned(grid, env, owner, i)
    lx = np.log10(grid[m])
    mid = 10.0 ** (0.5 * (lx[0] + lx[-1]))
    return mid, float(np.interp(np.log10(mid), lx, env[m]))


def fit_exponent(grid, env, l_max: float = 8.0) -> tuple[float, float, float]:
    """Least squares slope of log L against log C over the envelope below `l_max`."""
    m = (env < l_max) & ~np.isnan(env)
    slope, intercept = np.polyfit(np.log10(grid[m]), np.log10(env[m]), 1)
    return float(slope), float(grid[m].min()), float(grid[m].max())


def above_convergence(data: dict, grid, env, owner, l_max: float = 8.0):
    """How far the frontier sits above the converged loss of the size that owns it.

    Sampled along the envelope, evenly in log compute, so it is a property of the
    frontier rather than of the six sizes we happen to draw.
    """
    cs = curves(data, "compute")
    plats = np.array([plateau(c) for c in cs])
    m = (env < l_max) & ~np.isnan(env) & (owner >= 0)
    r = env[m] / plats[owner[m]]
    return float(np.median(r)), float(r.min()), float(r.max())


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


def legend_html(labels) -> str:
    spans = "\n".join(
        f'<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true">'
        f'<line x1="1" y1="5" x2="29" y2="5" stroke="{c}" stroke-width="2.4"/>'
        f"</svg>{lab}</span>"
        for lab, c in zip(labels, RAMP))
    return ('<div class="cap-legend kap-legend">\n'
            '<span class="num">parameters <em>N</em></span>\n'
            f"{spans}\n</div>")


# ---------------------------------------------------------------- the two figures

Y_TICKS = (2.5, 3, 4, 6, 8, 10)
Y_LO, Y_HI = 2.3, 11.4

TOKENS_LO, TOKENS_HI = 3e5, 4e11
COMPUTE_LO, COMPUTE_HI = 1e-11, 3e1
TOKEN_TICKS = (1e6, 1e8, 1e10)
COMPUTE_TICKS = (1e-9, 1e-6, 1e-3, 1e0)

ARIA_TOKENS = ("Test loss against the number of tokens processed, log-log, six "
               "training curves digitized from Kaplan's Figure 2, for model sizes "
               "from about a thousand to about a billion parameters. Every curve "
               "starts flat at the initialization loss of 10.8 nats, drops in an "
               "S-shape and settles onto its own converged loss, from 6.0 for the "
               "smallest to 2.4 for the largest. The larger the model, the earlier "
               "its curve leaves the flat top and the further left it sits, so a "
               "larger model reaches any given loss after fewer tokens.")
ARIA_COMPUTE = ("Test loss against training compute in petaflop-days, log-log, the "
                "same six digitized training curves, each one shifted right of the "
                "next smaller size. Their lower envelope, taken over all fifteen "
                "recovered curves, is the compute-efficient frontier and is close to "
                "a straight line in log-log with slope minus 0.06. A diamond marks "
                "the point on each drawn curve at which that model size is the "
                "compute-optimal one, visibly above the curve's own plateau.")


def header(data: dict, kind: str, extra: list[str] | None = None) -> str:
    """The generated-file banner: what this draws, and where every number came from."""
    src = data["source"]
    out = [
        "<!-- Generated by scripts/kaplan_curves.py --write -- do not edit by hand.",
        "",
        f"     {kind}",
        "",
        f"     {src['figure']}, digitized from the published image",
        f"     {src['image']}",
        "     by scripts/kaplan_figure2.py -> results/kaplan_figure2.json.  Method:",
        "     per-colour pixel extraction, continuity-tracked; the loss and compute",
        "     axes and the model sizes are calibrated from the image's own gridlines,",
        "     ticks and colourbar labels.  Nothing here is evaluated from the paper's",
        "     fitted L(N, S_min): these are the curves the figure draws.",
        "",
        "     Caveats, from that script's docstring:",
        "       * the loss axis is linear in the original and one pixel is 0.026",
        "         nats, so loss quantization is not an issue; the x axes are log.",
        "       * N is read off the figure's own colourbar and carries about +-10 %.",
        "       * inside the dense bundle a curve is partly painted over by its",
        "         neighbours, so it is sampled more sparsely there.",
        "     Drawn: a 5-sample moving mean of the extraction, resampled onto 150",
        "     points even in log x.  No monotone projection and no fit.",
        "",
        f"     6 of the 15 recovered sizes, evenly spaced in log N: "
        f"{', '.join(label_of(c['n']) for c in [curves(data, 'tokens')[i] for i in PICK])}.",
    ]
    if extra:
        out += [""] + [f"     {line}" for line in extra]
    return "\n".join(out + ["-->"])


def tokens_svg(data: dict) -> str:
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
    cs = curves(data, "tokens")
    for i, col in zip(PICK, RAMP):
        x, y = resample(*clean(cs[i]))
        s.append(f'<path d="{polyline(x, y, sx, sy)}" fill="none" stroke="{col}" '
                 f'stroke-width="2.6" stroke-linecap="round"/>')
    s += ["</g>", "</svg>"]
    return "\n".join(s)


def compute_svg(data: dict) -> str:
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
                "</tspan> (PF-days)", "loss", COMPUTE_TICKS, Y_TICKS),
         '<g clip-path="url(#kc-box)">']
    cs = curves(data, "compute")
    for i, col in zip(PICK, RAMP):
        x, y = resample(*clean(cs[i]))
        s.append(f'<path d="{polyline(x, y, sx, sy)}" fill="none" stroke="{col}" '
                 f'stroke-width="2.6" stroke-linecap="round"/>')
    s.append("</g>")

    # The frontier, and the point on each drawn run where that run is the optimal one.
    # The group carries a marker colloquium can count (see scripts/_steps.py); the prose
    # step about compute-optimal training stopping short of convergence keeps the
    # matching explicit index, so the two land on the same click.
    grid, env, owner = envelope(data)
    # Drawn from where the smallest *drawn* size first takes over the envelope: left of
    # that the envelope belongs to sizes this figure does not plot, and a dashed line
    # under an empty stretch of plot reads as an error rather than as an envelope.
    start = grid[owned(grid, env, owner, PICK[0])][0]
    m = (grid >= start) & ~np.isnan(env)
    steps = Steps()
    s += [f'<g class="fragment"{steps.attr(1)}>',
          '<g clip-path="url(#kc-box)">',
          f'<path d="{polyline(grid[m], env[m], sx, sy)}" fill="none" '
          f'stroke="{FRONTIER}" stroke-width="2.6" stroke-dasharray="7 5" '
          f'stroke-linecap="round"/>',
          "</g>"]
    for i, col in zip(PICK, RAMP):
        c, l = optimum(grid, env, owner, i)
        s.append(f'<path d="{diamond(sx(c), sy(l))}" fill="{col}" stroke="{SURFACE}" '
                 f'stroke-width="1.6"/>')
    xs, ys = grid[m], env[m]
    p0 = (float(sx(xs[0])), float(sy(ys[0])))
    p1 = (float(sx(xs[-1])), float(sy(ys[-1])))
    # The label hugs the underside of the frontier near its left end: that wedge --
    # below the frontier, left of every plateau -- is the one empty part of the plot,
    # and above the line the label would have to cross five descending curves.
    s += [along(p0, p1, 0.30, "compute-efficient frontier", -46.0,
                "pf-red pf-strong"),
          "</g>", "</svg>"]
    return "\n".join(s)


def body(data: dict, panel: str, svg: str) -> str:
    labels = [label_of(c["n"]) for c in [curves(data, panel)[i] for i in PICK]]
    return legend_html(labels) + "\n\n" + svg


def write(data: dict) -> None:
    grid, env, owner = envelope(data)
    slope, c_lo, c_hi = fit_exponent(grid, env)
    med, lo, hi = above_convergence(data, grid, env, owner)
    FIG_TOKENS.write_text(
        header(data, "Loss against tokens processed (Kaplan Fig. 2, left panel).",
               ["The paper's loss axis is linear; this one is log, so that the",
                "horizontal shift between the plateaus reads as a shift.  Recovered",
                "converged losses: "
                + ", ".join(f"{label_of(curves(data, 'tokens')[i]['n'])} -> "
                            f"{plateau(curves(data, 'tokens')[i]):.2f}"
                            for i in PICK) + "."])
        + "\n\n" + body(data, "tokens", tokens_svg(data)) + "\n")
    FIG_COMPUTE.write_text(
        header(data, "Loss against compute in PF-days, with the compute-efficient\n"
                     "     frontier revealed on the first beat (Kaplan Fig. 2, right "
                     "panel).",
               ["1 PF-day = 8.64e19 FLOPs; the paper's own x axis is in PF-days, and",
                "the deck's C = 6ND is in FLOPs.",
                "",
                "The frontier is the pointwise lower envelope of all 15 recovered",
                "curves, drawn from where the smallest plotted size takes it over.",
                f"Fitting log L against log C below L = 8 ({c_lo:.1e} .. {c_hi:.1e}",
                f"PF-days) gives L ~ C^{slope:+.3f}; the paper's own fit to its",
                "compute-efficient frontier, eq. (1.3), is C^-0.050.",
                "",
                "Along that envelope the loss sits a median "
                f"{100 * (med - 1):.0f} % above the converged",
                f"loss of the size that owns it ({100 * (lo - 1):.0f} to "
                f"{100 * (hi - 1):.0f} % over the frontier), which is what",
                "the slide's undertraining beat quotes.  The diamonds sit at the",
                "middle, in log compute, of each drawn size's own stretch."])
        + "\n\n" + body(data, "compute", compute_svg(data)) + "\n")
    print(f"wrote {FIG_TOKENS.relative_to(ROOT)} and {FIG_COMPUTE.relative_to(ROOT)}")


def report(data: dict) -> None:
    tok, com = curves(data, "tokens"), curves(data, "compute")
    print(f"{len(tok)} curves recovered per panel; drawing indices {PICK}")
    print("  drawn curves (tokens panel / compute panel):")
    for i, col in zip(PICK, RAMP):
        xt, yt = clean(tok[i])
        xc, yc = clean(com[i])
        print(f"   idx{i:2d} N = {tok[i]['n']:9.2e} '{label_of(tok[i]['n']):>5s}' {col}"
              f"  D {xt[0]:8.1e}..{xt[-1]:8.1e}  C {xc[0]:8.1e}..{xc[-1]:8.1e} PF-d"
              f"  L {yt.max():5.2f}->{yt[-1]:5.2f} (plateau {plateau(tok[i]):5.3f}, "
              f"compute panel {plateau(com[i]):5.3f})")
    dis = max(abs(plateau(t) - plateau(c)) for t, c in zip(tok, com))
    print(f"  panels agree on every converged loss to {dis:.3f} nats")
    print(f"  tokens axis needs {min(min(c['x']) for c in tok):.1e} .. "
          f"{max(max(c['x']) for c in tok):.1e} "
          f"(drawn {TOKENS_LO:.0e} .. {TOKENS_HI:.0e})")
    print(f"  compute axis needs {min(min(c['x']) for c in com):.1e} .. "
          f"{max(max(c['x']) for c in com):.1e} PF-days "
          f"(drawn {COMPUTE_LO:.0e} .. {COMPUTE_HI:.0e})")

    grid, env, owner = envelope(data)
    print("\ncompute-efficient frontier = lower envelope of all "
          f"{len(com)} digitized curves:")
    for l_max in (9.5, 8.0, 6.0, 5.0):
        slope, c_lo, c_hi = fit_exponent(grid, env, l_max)
        print(f"  fit below L = {l_max:4.1f} ({c_lo:8.1e} .. {c_hi:8.1e} PF-days): "
              f"L ~ C^{slope:+.4f}")
    print("  (the paper's own fit to its frontier, eq. 1.3, is C^-0.050)")
    med, lo, hi = above_convergence(data, grid, env, owner)
    print(f"  along the envelope, loss / converged loss of the owning size: "
          f"median {med:.3f}, range {lo:.3f}..{hi:.3f}")

    print("\nwhere each size owns the frontier (the diamonds, on the drawn six):")
    for i in range(len(com)):
        m = owned(grid, env, owner, i)
        if m.sum() < 3:
            print(f"   idx{i:2d} N = {com[i]['n']:9.2e}  owns nothing")
            continue
        c, l = optimum(grid, env, owner, i)
        p = plateau(com[i])
        print(f"  {'*' if i in PICK else ' '}idx{i:2d} N = {com[i]['n']:9.2e}  owns "
              f"{grid[m][0]:8.1e}..{grid[m][-1]:8.1e} PF-days, mid C = {c:8.1e} at "
              f"L = {l:5.3f} = {100 * (l / p - 1):4.1f} % above its plateau {p:5.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write figures/kaplan-tokens-fig.md and "
                         "figures/kaplan-compute-fig.md")
    args = ap.parse_args()
    data = load()
    report(data)
    if args.write:
        write(data)


if __name__ == "__main__":
    main()
