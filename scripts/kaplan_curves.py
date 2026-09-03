"""The two "Scaling laws" slides: Kaplan's Figure 2, digitized off the figure itself.

Kaplan et al. [kaplan2020scaling] never published the runs behind Figure 2, and its
fitted L(N, S_min) is a smooth power law that misses what the figure actually shows --
the flat stretch at the initialization loss, the S-shaped drop, the noise, the plateau
each run settles onto.  So nothing here is evaluated from a law.  Every line comes from
`scripts/kaplan_figure2.py`, which reads the pixels of the published image and writes
`results/kaplan_figure2.json`: 20 curves per panel, each `{n, rgb, x, loss}`, plus a
`merge` record per curve carrying the log-x shift that maps the compute panel onto the
tokens panel and how far apart the two panels came out.  Both panels hold the *same*
samples -- the extractor fills each panel's occlusion gaps from the other one -- so
`panel_curves` reuses that shift rather than treating the panels as two data sets.
Read that script's docstring for the calibration and the caveats; the two that matter
for what is drawn here are that N is read off the figure's own colourbar and carries
about +-10 %, and that where a curve is painted over inside the dense bundle the colour
that survives is a composite, which reads as the colour of a *third* curve between the
two.  That last one is why what follows is necessary.

    figures/kaplan-tokens-fig.md    loss against tokens processed D   (Fig. 2, left)
    figures/kaplan-compute-fig.md   loss against compute C, in FLOPs  (Fig. 2, right)

Two structural facts about that figure are enforced on the digitized curves before
they are drawn, because the extraction cannot be made good enough to satisfy them on
its own.  In the published left panel the twenty runs are an *ordered, nested fan*:
each one starts at the same initialization loss, a larger model leaves the flat top
earlier, and no two curves ever cross.  Inside the dense bundle the extraction does
not reproduce that -- a stroke there is painted over in part, so the reading wanders
a fraction of a nat off the line and back -- and drawn raw the family visibly tangles
and steps.  So the family is projected onto those two facts: monotone non-increasing
in D, and ordered by N at every D.  What that projection is, exactly, and how far it
has to move each curve, is in `projected` and in the report; it is a projection of the
measurement, never a fit, and the paper's L(N, S_min) still appears nowhere.

All twenty sizes are drawn -- the picture is "one hue, ordered by N", and it says that
better with the whole family in it than with a subset.  Only six of them are *labelled*,
the six roughly evenly spaced in log N that carry the deck's six-step blue ramp; the
other fourteen take a colour interpolated on that same ramp at their own log N, and a
thinner stroke, so they fill the family in without competing with the six the legend
names.  The compute panel also carries the **compute-efficient frontier**, computed as
the true lower envelope of the digitized curves -- the pointwise minimum over all
twenty, not a fitted line -- with a diamond on each labelled run at the middle of the
stretch of compute over which that run *is* the envelope.  It is drawn in the deck's
muted grey, as a guide rather than as a measurement to read numbers off, and its in-plot
label quotes the exponent *the paper* fits to its own frontier, eq. (1.3)'s C^-0.050,
not the -0.059 we measure off the envelope: the label names Kaplan's claim, and the gap
between the two numbers is a digitization artefact, not a finding.  (Cleaning up the
extraction did not close that gap, and was never going to: the crossings that made this
figure unusable were in the middle of the drop, which the lower envelope barely touches.
Fitting below L = 8, the exponent was -0.0595 with the tangled fifteen-curve extraction
and is -0.0595 with the ordered twenty-curve one -- unchanged to four decimals.  The gap
to eq. (1.3) is somewhere else: most likely in the paper fitting its frontier over a
narrower range of C than the eight decades available here.)  The fitted L(N, C) appears
nowhere.

The compute axis is in FLOPs, not the paper's PF-days: everything else in the deck
counts C = 6ND in FLOPs, and a reader should not have to carry a unit conversion from
one slide to the next.  All the arithmetic below stays in PF-days, which is what the
digitization records; the conversion happens once, at the axis (`to_flops`).

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

# Which of the twenty recovered sizes the *legend* names.  Indices into the JSON's curve
# list, which is sorted by N: these six are as evenly spaced in log N as the recovered
# set allows (gaps of 1.17, 1.25, 1.25, 1.25, 0.94 decades) and span 5.8 of its 6.3
# decades.  Six keys is already the most the legend fits on one line (see .kap-legend in
# assets/slides.css), so the other fourteen sizes are drawn but not named.  The very
# smallest recovered curve (N ~ 5.6e2) is one of the unnamed ones: the figure paints it
# over for most of its drop, so it is the curve the extraction knows least about.
PICK = (1, 5, 8, 13, 16, 19)
# Cheap to expensive, small to large: the deck's single-hue ramp, since N is *ordered*.
# The six stops are pinned to the six PICK sizes and the fourteen others interpolated
# between them in log N -- see `ramp_colours` -- so the legend stays exactly true.
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#256abf", "#184f95", "#0d366b")
SURFACE = "#fcfcfb"
# The frontier is a guide, not a measurement: grey, like `.pf-guide` and the axes, and
# not the deck's red, which is reserved for the one quantity a slide is *about*.  This is
# --colloquium-muted from the theme, written out because the SVG carries its own strokes.
FRONTIER = "#6b7280"
LABELLED_W, PLAIN_W = 2.6, 1.7   # stroke widths: the six the legend names, and the rest
# The exponent Kaplan et al. fit to their *own* compute-efficient frontier, eq. (1.3):
# L = (C_min / 2.3e8 PF-days)^-0.050.  This, not the -0.059 the report measures off the
# digitized envelope, is what the in-plot label quotes: the label attributes a claim to
# the paper, and the 0.009 between the two is an artefact of reading pixels.
PAPER_SLOPE = "-0.050"
# Where the envelope is a power law rather than a run-in from the initialization loss.
# One number, used three times: the stretch of the frontier that gets drawn, the stretch
# the exponent is fitted over, and the stretch the undertraining figure is measured over.
L_FIT = 8.0
# Half a pixel of the source figure (one pixel is 0.026 nats).  Two sizes closer than
# this on the frontier are a tie the digitization cannot resolve; see `owned`.
TIE = 0.013

SMOOTH = 5                  # moving mean, in samples, over the digitized loss
DEKINK = 9                  # moving mean, in grid points, after the ordering swap
GRID_D = 900                # family grid, even in log D: ~0.006 decades per point
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
# 414 units cleared that bar but only just: the x-axis title landed a few pixels off the
# inline footnote under the figure.  H and YB come down by 22 together, which keeps the
# 22 units of padding under the axis title and takes 5 % off the rendered height (the
# figure is `width: 100%`, so height scales with H/W) -- enough clearance, and small
# enough that the axis type still renders at the ~1 unit = 1 px the deck is tuned for.
W, H = 680, 392
YT, YB = 14, 308          # plot box, top and bottom
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

    Per-curve processing only: a `SMOOTH`-sample moving mean of the loss, which takes
    out the single-pixel jitter of the extraction (one pixel is 0.026 nats).  This is
    the input to `projected`, which is where the two structural constraints are
    imposed; on its own this leaves the family tangled.
    """
    x = np.asarray(curve["x"], float)
    y = np.asarray(curve["loss"], float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if SMOOTH > 1 and len(y) > SMOOTH:
        pad = SMOOTH // 2
        y = np.convolve(np.pad(y, pad, mode="edge"), np.ones(SMOOTH) / SMOOTH, "valid")
    return x, y


def smooth_rows(m: np.ndarray, w: int) -> np.ndarray:
    """A `w`-point moving mean along each row of `m`, edge-padded."""
    if w <= 1:
        return m
    k = np.ones(w) / w
    return np.vstack([np.convolve(np.pad(r, w // 2, mode="edge"), k, "valid")
                      for r in m])


def projected(data: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """The twenty runs on one grid in D, projected onto the source figure's structure.

    Returns (log10 D grid, 20 x len(grid) losses ordered by N, a report dict).

    The projection is three steps, in this order, and every one of them is a map
    applied to the measurement rather than a curve fitted to it:

      order in N   At each D the twenty readings are *reassigned* to the twenty
                   sizes in rank order -- largest loss to the smallest model.  This is
                   a permutation of the measured values at that D: no value is
                   invented, moved in x, or altered in magnitude, and the set of
                   twenty losses drawn at any D is exactly the set extracted there.
                   Where the extraction is already ordered, which is most of the plot,
                   it is the identity.  The report counts where it is not.

                   A rank sort is the right projection here rather than merely a
                   convenient one, because of what the extraction's error actually is.
                   Inside the bundle twenty strokes are packed into eighty rows, and
                   where one is painted over, the colour that survives is a composite
                   -- and this colourmap is locally straight in RGB, so a composite of
                   two strokes reads as the colour of a *third* one lying between them.
                   The failure is therefore not that a stroke is found in the wrong
                   place: it is that the right places are handed to the wrong sizes.
                   At D = 1.0e8, for instance, the twenty readings are a perfectly
                   sensible fan but five of them are shuffled, three of those by about
                   a nat.  Sorting is exactly the inverse of a shuffle.
      de-kink      A `DEKINK`-point moving mean along D.  Swapping two curves that
                   touch leaves each of them with a corner; this rounds it off.  A
                   moving mean is order-preserving (it is the same linear map on every
                   row), so it cannot undo the step above.
      monotone     Cumulative minimum along increasing D.  The published curves never
                   go back up; ours do, by up to 0.6 nats, on the three runs whose two
                   panels disagree by a couple of pixels over the knee of their drop,
                   which puts two slightly offset readings of the same point side by
                   side.  A cumulative minimum is also order-preserving, so the
                   ordering survives it.

    Only the *tokens* structure is imposed, i.e. loss as a function of D.  The compute
    panel is the same runs against C = 6ND and there the curves genuinely do cross --
    a small model has converged while a large one is still at the initialization loss,
    which is exactly what makes the compute-efficient frontier change hands -- so
    ordering must not be, and is not, imposed on it.  `panel_curves` maps this result
    onto whichever x axis a panel uses.

    The grid spans the D that *every* run covers, so that the rank reassignment always
    ranks twenty readings and never a varying subset.  That trims about 0.1 decades
    off each end of a 5.5-decade extraction, in the flat top and on the plateau where
    nothing happens.
    """
    cs = curves(data, "tokens")
    xs, ys = zip(*(clean(c) for c in cs))
    lo = max(x[0] for x in xs)
    hi = min(x[-1] for x in xs)
    lg = np.linspace(np.log10(lo), np.log10(hi), GRID_D)
    raw = np.vstack([np.interp(lg, np.log10(x), y) for x, y in zip(xs, ys)])

    ordered = -np.sort(-raw, axis=0)      # row 0 = smallest N = highest loss
    swap = np.abs(ordered - raw)
    rep = {
        # Two counts, because the first one alone overstates the case: most of the
        # points where the extraction is not exactly ordered are ties of one or two
        # pixels in the flat top and on the plateaus, where twenty curves sit inside
        # a band 0.05 nats wide and their order is below the figure's resolution.  The
        # second count is the disorder that is actually visible: more than 0.05 nats,
        # two pixels, which is the width of a stroke in the source figure.
        "crossing_points": int((swap > 1e-9).any(0).sum()),
        "crossing_visible": int((swap > 0.05).any(0).sum()),
        "crossing_worst": float(swap.max()),
        "rise_worst": float(np.maximum(np.diff(raw, axis=1), 0).max()),
    }
    m = np.minimum.accumulate(smooth_rows(ordered, DEKINK), axis=1)
    rep["moved_per_curve"] = [float(np.abs(m[i] - raw[i]).max()) for i in range(len(cs))]
    rep["moved_median"] = [float(np.median(np.abs(m[i] - raw[i]))) for i in range(len(cs))]
    return lg, m, rep


def panel_curves(data: dict, panel: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The projected family on one panel's own x axis, as twenty (x, loss) pairs.

    Both panels of the JSON hold the *same* samples of the same runs -- the extractor
    merges them on a fitted log-x shift per curve -- so the compute panel is this
    panel's D grid divided by that curve's shift, and no second projection is needed
    or wanted.
    """
    lg, m, _ = projected(data)
    if panel == "tokens":
        return [(10.0 ** lg, m[i]) for i in range(m.shape[0])]
    shifts = [rec["shift"] for rec in data["merge"]]
    return [(10.0 ** (lg - shifts[i]), m[i]) for i in range(m.shape[0])]


def resample(x: np.ndarray, y: np.ndarray, n: int = RESAMPLE):
    """The curve on `n` points even in log x: an SVG path needs no more than that."""
    lx = np.linspace(np.log10(x[0]), np.log10(x[-1]), n)
    return 10.0 ** lx, np.interp(lx, np.log10(x), y)


def curves(data: dict, panel: str) -> list[dict]:
    return data["curves"][panel]


def label_of(n: float) -> str:
    """A model size as a rounded label: 1.8k, 26k, 470k, 8.2M, 150M, 1.3B."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.0f}{suf}" if v >= 10 else f"{v:.1f}{suf}"
    return f"{n:.0f}"


def to_flops(c_pf):
    """Compute in PF-days -> compute in FLOPs.

    The digitization is in the paper's own x units; the deck's C = 6ND is in FLOPs, and
    the axis is in FLOPs.  Everything upstream of the axis stays in PF-days so that the
    report's numbers can be checked against the published figure directly.
    """
    return np.asarray(c_pf, float) * PF_DAY


def ramp_colours(ns: np.ndarray) -> list[str]:
    """One colour per recovered size, on the single-hue ramp, ordered by N.

    The six `PICK` sizes are the ramp's control points and get their `RAMP` stop exactly,
    so the legend keeps meaning what it says; every other size is a linear blend of the
    two stops it falls between, in log N.  Blending straight in sRGB is fine here only
    because the stops share a hue and differ mostly in lightness -- the point of the ramp
    is that it reads as ordered, and interpolating a single hue cannot break that order.
    The one size below the first stop (N ~ 5.6e2) clamps to the lightest colour.
    """
    anchors = np.log10(np.asarray(ns, float)[list(PICK)])
    stops = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in RAMP], float)
    out = []
    for n in ns:
        t = float(np.clip(np.log10(n), anchors[0], anchors[-1]))
        rgb = [int(round(np.interp(t, anchors, stops[:, k]))) for k in range(3)]
        out.append("#%02x%02x%02x" % tuple(rgb))
    return out


# ---------------------------------------------------------------- the envelope


def envelope(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The compute-efficient frontier: the pointwise lower envelope of the 20 curves.

    On a log grid in compute, every curve of `panel_curves("compute")` is interpolated
    (NaN outside its own range) and the minimum taken.  That is all: no law, no fit, no
    smoothing of the envelope itself.  The curves it envelopes are the projected ones,
    so the frontier inherits that projection and nothing else.  Returns (compute in
    PF-days, envelope loss, index of the strict argmin at each grid point, and the
    whole 20 x GRID matrix -- `owned` needs the matrix, not just the argmin, because
    two sizes can share the frontier to within the figure's own pixel).
    """
    cs = panel_curves(data, "compute")
    lo = min(x[0] for x, _ in cs)
    hi = max(x[-1] for x, _ in cs)
    grid = np.logspace(np.log10(lo), np.log10(hi), GRID)
    m = np.vstack([np.interp(np.log10(grid), np.log10(x), y,
                             left=np.nan, right=np.nan) for x, y in cs])
    with np.errstate(invalid="ignore"):
        env = np.nanmin(m, axis=0)
    owner = np.array([int(np.nanargmin(col)) if not np.all(np.isnan(col)) else -1
                      for col in m.T])
    return grid, env, owner, m


def plateaus(data: dict) -> np.ndarray:
    """The converged loss of each run: the lowest loss it reaches, ordered by N.

    Read off the projected family, where it is simply the last point of the curve --
    the projection is monotone, so the minimum is at the largest D.
    """
    _, m, _ = projected(data)
    return m[:, -1]


def owned(grid, env, mat, i: int, l_max: float = 9.5):
    """The stretch of the frontier that curve `i` owns, as a boolean mask.

    Restricted to the part of the envelope that has left the initialization loss: above
    L ~ 9.5 every run is still flat at ln(50257) = 10.8 and "the cheapest model" is
    meaningless.

    "Owns" means within TIE of the envelope, not exactly equal to it.  A strict argmin
    is the wrong test on digitized data: three of the smallest sizes have converged
    losses within half a nat of each other, so which of them is cheapest over a given
    stretch is settled by hundredths of a nat, and at TIE = 0 the 1.8k run -- one of the
    six the legend names -- loses its whole stretch to a margin of 0.0013 nats, a
    twentieth of one pixel of the source figure.  TIE is half a pixel.
    """
    with np.errstate(invalid="ignore"):
        near = (mat[i] - env) <= TIE
    return near & (env < l_max) & ~np.isnan(env) & ~np.isnan(mat[i])


def optimum(grid, env, mat, i: int) -> tuple[float, float]:
    """Where model `i` is the compute-optimal one: (compute in PF-days, loss).

    A size owns a *stretch* of the frontier, not a point, so we take the middle of that
    stretch in log compute.  The endpoints are the two switch-overs to the neighbouring
    sizes and are the noisiest part of the extraction, which is the other reason not to
    quote them.
    """
    m = owned(grid, env, mat, i)
    lx = np.log10(grid[m])
    mid = 10.0 ** (0.5 * (lx[0] + lx[-1]))
    return mid, float(np.interp(np.log10(mid), lx, env[m]))


def fit_exponent(grid, env, l_max: float = L_FIT) -> tuple[float, float, float]:
    """Least squares slope of log L against log C over the envelope below `l_max`."""
    m = (env < l_max) & ~np.isnan(env)
    slope, intercept = np.polyfit(np.log10(grid[m]), np.log10(env[m]), 1)
    return float(slope), float(grid[m].min()), float(grid[m].max())


def above_convergence(data: dict, grid, env, owner, l_max: float = L_FIT):
    """How far the frontier sits above the converged loss of the size that owns it.

    Sampled along the envelope, evenly in log compute, so it is a property of the
    frontier rather than of the six sizes we happen to draw.
    """
    plats = plateaus(data)
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


def named_exponent(x: float, y: float, name: str, base: str, exp: str,
                   lead: float = 26.0) -> list[str]:
    """A curve's name, and under it the exponent, as two separate labels.

    Two elements rather than one two-line block because they are different kinds of
    type.  The name is a caption and stays bold; the exponent is set exactly as every
    other exponent in the deck is -- muted, horizontal, and nothing but `base^exp`, no
    `L =` or `slope =` in front of it -- so the room reads all of them as one family.
    Neither is rotated: the frontier is the shallowest line in the deck (slope -0.05 in
    log-log, about 20 degrees on screen once the axes are scaled) and type set at that
    angle reads as a mistake rather than as a label.  Both are parked in the empty wedge
    under the frontier, the one region of the plot that is empty by construction, since
    nothing can sit below its own lower envelope.
    """
    return [f'<text class="pf-muted pf-strong pf-small" x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="middle">{name}</text>',
            f'<text class="pf-muted pf-small" x="{x:.1f}" y="{y + lead:.1f}" '
            f'text-anchor="middle">{base}'
            f'<tspan dx="1" dy="-9" font-size="0.72em">{exp}</tspan></text>']


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
TOKEN_TICKS = (1e6, 1e8, 1e10)
# In FLOPs, not PF-days.  The recovered curves span 2.3e9 .. 5.8e20 FLOPs, so the box is
# opened half a decade either side of that and ticked every third decade: 13 decades of x
# in 568 units leaves ~44 units per decade, which is just wide enough for `10^k` labels
# not to crowd, and every tick then sits comfortably inside the box rather than on its
# corner (a label centred on X1 would run off the viewBox).
COMPUTE_LO, COMPUTE_HI = 3e8, 3e21
COMPUTE_TICKS = (1e9, 1e12, 1e15, 1e18, 1e21)

ARIA_TOKENS = ("Test loss against the number of tokens processed, log-log, twenty "
               "training curves digitized from Kaplan's Figure 2, for model sizes "
               "from about five hundred to about a billion parameters, six of them "
               "labelled. Every curve "
               "starts flat at the initialization loss of 10.8 nats, drops in an "
               "S-shape and settles onto its own converged loss, from 6.4 for the "
               "smallest to 2.4 for the largest. The larger the model, the earlier "
               "its curve leaves the flat top and the further left it sits, so a "
               "larger model reaches any given loss after fewer tokens.")
ARIA_COMPUTE = ("Test loss against training compute in FLOPs, log-log, the "
                "same twenty digitized training curves, each one shifted right of "
                "the next smaller size. Their lower envelope is the "
                "compute-efficient frontier, drawn as a grey dashed line, and is "
                "close to a straight line in log-log; the paper fits it as loss "
                "proportional to compute to the minus 0.050. A diamond marks "
                "the point on each labelled curve at which that model size is the "
                "compute-optimal one, visibly above the curve's own plateau.")


def max_gap(data: dict, l_max: float = 10.4) -> float:
    """Widest gap in log10 D, over all curves, on the part below `l_max`.

    Above 10.4 nats every run is still flat at the initialization loss and a gap there
    costs nothing; below it, this is the honest bound on how much of a drawn curve is
    interpolation between samples rather than samples.
    """
    worst = 0.0
    for c in curves(data, "tokens"):
        x, y = clean(c)
        lx = np.log10(x)
        for k in np.where(y[:-1] < l_max)[0]:
            worst = max(worst, float(lx[k + 1] - lx[k]))
    return worst


def apart(data: dict) -> tuple[float, int]:
    """Cross-panel disagreement: (worst, how many curves under 0.03 decades).

    Straight out of the extractor's own fit -- see `disagreement` in
    scripts/kaplan_figure2.py for what the number means and why it is in decades of D
    rather than in nats.
    """
    v = [rec["apart_decades"] for rec in data["merge"]]
    return float(max(v)), int(sum(1 for a in v if a < 0.03))


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
        "     per-colour pixel extraction at the exact stroke colour, nearest",
        "     reference wins; per column a median and a window around a running",
        "     median of those medians; and the left panel's occlusion gaps filled",
        "     from the right panel, which draws the same runs against C = 6ND and so",
        "     is the same curve slid along log x, on a shift fitted per curve.  The",
        "     loss and compute axes and the model sizes are calibrated from the",
        "     image's own gridlines, ticks and colourbar labels.  Nothing here is",
        "     evaluated from the paper's fitted L(N, S_min): these are the curves the",
        "     figure draws.",
        "",
        f"     The figure has {len(curves(data, 'tokens'))} curves, not the 15 an "
        "earlier version of this",
        "     digitization found: it deduplicated candidate colours that landed within",
        "     0.27 decades of N of each other on the colourbar, and the figure's sizes",
        "     are spaced 0.1 to 0.6 decades apart, so five real curves were dropped",
        "     and their pixels claimed by whichever survivor was nearest in colour.",
        "",
        "     Caveats, from that script's docstring:",
        "       * the loss axis is linear in the original and one pixel is 0.026",
        "         nats, so loss quantization is not an issue; the x axes are log.",
        "       * N is read off the figure's own colourbar and carries about +-10 %.",
        "       * the colour tolerance cannot be loosened to chase a curve into the",
        "         dense bundle: this colourmap is locally straight in RGB, so where",
        "         two curves overlap the blend is nearer the reference colour of the",
        "         curve between them than that curve's own anti-aliasing is.  A",
        "         stretch where a curve is fully painted over is left as a gap.",
        "       * after the gap filling, the widest gap anywhere below L = 10.4 is",
        f"         {max_gap(data):.3f} decades, five pixels of the left panel; every gap "
        "wider than",
        "         0.1 decades is in the flat top, where nothing happens.",
        "       * the two panels place the same point of the same run within "
        f"{apart(data)[0]:.2f}",
        f"         decades of D ({apart(data)[1]:.0f} of 20 curves within 0.03), which "
        "is the one",
        "         self-check available without the original data.",
        "",
        "     PROCESSING, before drawing -- this is not raw digitization:",
        "       1. a 5-sample moving mean per curve, against pixel jitter;",
        "       2. the twenty curves put on one grid in D and, at each D, the",
        "          twenty readings REASSIGNED to the twenty sizes in rank order,",
        "          largest loss to the smallest model.  A permutation of the measured",
        "          values, not a fit; it enforces the published figure's ordering, in",
        "          which no two of the twenty ever cross.  Inside the dense bundle the",
        "          extraction's error is precisely a shuffle -- where a stroke is",
        "          painted over, the surviving composite colour reads as the colour of",
        "          a third curve between the two -- so the right positions are found",
        "          and handed to the wrong sizes, and a rank sort inverts that;",
        "       3. a 7-point moving mean along D, to round off the corners step 2",
        "          leaves where two curves are swapped;",
        "       4. a cumulative minimum along D, enforcing that a run's loss never",
        "          goes back up, which in the published figure it never does.",
        "     Steps 2-4 are order-preserving maps of the measurement, and the flat",
        "     top, the S-shaped drop and the distinct plateaus all survive them.  They",
        "     are imposed on loss-against-D only: against C the twenty curves DO",
        "     cross, which is what makes the frontier change hands.  Resampled onto",
        "     150 points even in log x for the path.",
        "",
        f"     All {len(curves(data, 'tokens'))} recovered sizes are drawn.  The "
        f"{len(PICK)} the legend names, evenly spaced in",
        f"     log N, carry the deck's blue ramp and a {LABELLED_W}-unit stroke:",
        "     "
        + ", ".join(label_of(c["n"])
                    for c in [curves(data, "tokens")[i] for i in PICK]) + ".",
        "     The others take a colour interpolated on that same ramp at their own",
        f"     log N and a {PLAIN_W}-unit stroke, so the plot reads as one hue ordered",
        "     by N without the unnamed curves competing with the named ones.",
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
    fam = panel_curves(data, "tokens")
    cols = ramp_colours(np.array([c["n"] for c in cs]))
    # Unlabelled first, then the six the legend names on top of them.  After `projected`
    # the family is nested and nothing crosses, so this is only about weight: the six
    # that have a key should be the ones that read on top where the bundle is dense.
    for group in (False, True):
        for i in range(len(cs)):
            if (i in PICK) != group:
                continue
            x, y = resample(*fam[i])
            s.append(f'<path d="{polyline(x, y, sx, sy)}" fill="none" '
                     f'stroke="{cols[i]}" '
                     f'stroke-width="{LABELLED_W if group else PLAIN_W}" '
                     f'stroke-linecap="round"/>')
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
         *frame(sx, sy, 'compute <tspan class="pf-var">C</tspan>', "loss",
                COMPUTE_TICKS, Y_TICKS),
         '<g clip-path="url(#kc-box)">']
    cs = curves(data, "compute")
    fam = panel_curves(data, "compute")
    cols = ramp_colours(np.array([c["n"] for c in cs]))
    for group in (False, True):
        for i in range(len(cs)):
            if (i in PICK) != group:
                continue
            x, y = resample(*fam[i])
            s.append(f'<path d="{polyline(to_flops(x), y, sx, sy)}" fill="none" '
                     f'stroke="{cols[i]}" '
                     f'stroke-width="{LABELLED_W if group else PLAIN_W}" '
                     f'stroke-linecap="round"/>')
    s.append("</g>")

    # The frontier, and the point on each drawn run where that run is the optimal one.
    # The group carries a marker colloquium can count (see scripts/_steps.py); the prose
    # step about compute-optimal training stopping short of convergence keeps the
    # matching explicit index, so the two land on the same click.
    grid, env, owner, mat = envelope(data)
    # Drawn over the stretch below L = L_FIT, which is exactly the stretch the header's
    # power-law fit covers.  Above it the envelope is still climbing towards the
    # initialization loss that every run shares, where "the cheapest model for this
    # budget" means nothing, and the line would be laid over the dense bundle at the top
    # of the plot rather than under it.
    m = (env < L_FIT) & ~np.isnan(env)
    steps = Steps()
    s += [f'<g class="fragment"{steps.attr(1)}>',
          '<g clip-path="url(#kc-box)">',
          f'<path d="{polyline(to_flops(grid[m]), env[m], sx, sy)}" fill="none" '
          f'stroke="{FRONTIER}" stroke-width="2.6" stroke-dasharray="7 5" '
          f'stroke-linecap="round"/>',
          "</g>"]
    for i in PICK:
        c, l = optimum(grid, env, mat, i)
        s.append(f'<path d="{diamond(sx(to_flops(c)), sy(l))}" fill="{cols[i]}" '
                 f'stroke="{SURFACE}" stroke-width="1.6"/>')
    # The label sits horizontally inside the wedge under the frontier, which is empty by
    # construction -- nothing can lie below its own lower envelope -- and, at these
    # coordinates, empty with room to spare: the frontier passes about 50 units above the
    # block's right-hand end and the x axis about 40 below its baseline.  Further right
    # the wedge closes as the frontier descends; further left the label drifts away from
    # the line it names.  The exponent is the paper's, eq. (1.3), not the one the report
    # measures off these pixels: see PAPER_SLOPE.
    s += [*named_exponent(300.0, 235.0, "compute-efficient frontier",
                          '<tspan class="pf-var">C</tspan>', PAPER_SLOPE),
          "</g>", "</svg>"]
    return "\n".join(s)


def body(data: dict, panel: str, svg: str) -> str:
    labels = [label_of(c["n"]) for c in [curves(data, panel)[i] for i in PICK]]
    return legend_html(labels) + "\n\n" + svg


def write(data: dict) -> None:
    grid, env, owner, mat = envelope(data)
    slope, c_lo, c_hi = fit_exponent(grid, env)
    med, lo, hi = above_convergence(data, grid, env, owner)
    FIG_TOKENS.write_text(
        header(data, "Loss against tokens processed (Kaplan Fig. 2, left panel).",
               ["The paper's loss axis is linear; this one is log, so that the",
                "horizontal shift between the plateaus reads as a shift.  Recovered",
                "converged losses, for the six the legend names: "
                + ", ".join(f"{label_of(curves(data, 'tokens')[i]['n'])} -> "
                            f"{plateaus(data)[i]:.2f}"
                            for i in PICK) + ".",
                "All twenty, ordered by N: "
                + ", ".join(f"{v:.2f}" for v in plateaus(data)) + "."])
        + "\n\n" + body(data, "tokens", tokens_svg(data)) + "\n")
    FIG_COMPUTE.write_text(
        header(data, "Loss against compute in FLOPs, with the compute-efficient\n"
                     "     frontier revealed on the first beat (Kaplan Fig. 2, right "
                     "panel).",
               ["The x axis is in FLOPs, like the deck's C = 6ND everywhere else; the",
                "paper's own axis is in PF-days and the digitization is in PF-days, so",
                "every number below is in PF-days and the axis multiplies by",
                "1 PF-day = 8.64e19 FLOPs.  Drawn range in FLOPs: "
                f"{COMPUTE_LO:.0e} .. {COMPUTE_HI:.0e},",
                "ticked every third decade ("
                + ", ".join(f"{t:.0e}" for t in COMPUTE_TICKS) + ").",
                "",
                "The frontier is the pointwise lower envelope of all 20 recovered",
                f"curves, drawn (in grey, dashed) over the stretch below L = {L_FIT:g},",
                "which is where it has left the shared initialization loss and become",
                f"a power law.  Fitting log L against log C there ({c_lo:.1e} ..",
                f"{c_hi:.1e} PF-days) gives L ~ C^{slope:+.3f}; the label in the plot",
                "instead quotes the paper's own fit to its own compute-efficient",
                f"frontier, eq. (1.3), C^{PAPER_SLOPE}, since the label is attributing",
                "a claim to Kaplan et al.",
                "",
                "Along that envelope the loss sits a median "
                f"{100 * (med - 1):.0f} % above the converged",
                f"loss of the size that owns it ({100 * (lo - 1):.0f} to "
                f"{100 * (hi - 1):.0f} % over the frontier), which is what",
                "the slide's undertraining beat quotes.  The diamonds sit at the",
                "middle, in log compute, of each labelled size's own stretch."])
        + "\n\n" + body(data, "compute", compute_svg(data)) + "\n")
    print(f"wrote {FIG_TOKENS.relative_to(ROOT)} and {FIG_COMPUTE.relative_to(ROOT)}")


def report(data: dict) -> None:
    tok, com = curves(data, "tokens"), curves(data, "compute")
    lg, fam, rep = projected(data)
    plats = plateaus(data)
    print(f"{len(tok)} curves recovered per panel, all drawn; the legend names {PICK}")
    print("  labelled curves, after the structural projection:")
    for i, col in zip(PICK, RAMP):
        xt, yt = panel_curves(data, "tokens")[i]
        xc, yc = panel_curves(data, "compute")[i]
        print(f"   idx{i:2d} N = {tok[i]['n']:9.2e} '{label_of(tok[i]['n']):>5s}' {col}"
              f"  D {xt[0]:8.1e}..{xt[-1]:8.1e}  C {xc[0]:8.1e}..{xc[-1]:8.1e} PF-d"
              f"  L {yt.max():5.2f}->{yt[-1]:5.2f} (plateau {plats[i]:5.3f})")
    print(f"  tokens axis needs {min(min(c['x']) for c in tok):.1e} .. "
          f"{max(max(c['x']) for c in tok):.1e} "
          f"(drawn {TOKENS_LO:.0e} .. {TOKENS_HI:.0e})")
    c_need = (min(min(c["x"]) for c in com), max(max(c["x"]) for c in com))
    print(f"  compute axis needs {c_need[0]:.1e} .. {c_need[1]:.1e} PF-days "
          f"= {to_flops(c_need[0]):.1e} .. {to_flops(c_need[1]):.1e} FLOPs "
          f"(drawn {COMPUTE_LO:.0e} .. {COMPUTE_HI:.0e} FLOPs)")

    # How much work the projection had to do.  The interesting number is not that the
    # output is ordered -- it is by construction -- but how far the extraction was from
    # being ordered on its own, i.e. how much of the picture is measurement.
    print(f"\nstructural projection over {GRID_D} grid points in D "
          f"({10.0 ** lg[0]:.1e} .. {10.0 ** lg[-1]:.1e}):")
    print(f"  extraction out of order in N at {rep['crossing_points']:4d} of "
          f"{GRID_D} points ({100 * rep['crossing_points'] / GRID_D:.0f} %), of which "
          f"{rep['crossing_visible']} by more than a stroke width "
          f"({100 * rep['crossing_visible'] / GRID_D:.0f} %); "
          f"worst swap {rep['crossing_worst']:.2f} nats")
    print(f"  worst rise in D before the cumulative minimum: "
          f"{rep['rise_worst']:.2f} nats")
    print("  how far each curve moved (max / median over the grid), nats:")
    for i in range(len(tok)):
        print(f"   {'*' if i in PICK else ' '}idx{i:2d} N = {tok[i]['n']:9.2e}  "
              f"max {rep['moved_per_curve'][i]:5.3f}  "
              f"median {rep['moved_median'][i]:5.3f}  plateau {plats[i]:5.3f}")
    bad = [i for i in range(len(plats) - 1) if plats[i] <= plats[i + 1]]
    print(f"  plateaus strictly ordered in N: {'yes' if not bad else f'NO at {bad}'}")

    grid, env, owner, mat = envelope(data)
    print("\ncompute-efficient frontier = lower envelope of all "
          f"{len(com)} projected curves:")
    for l_max in (9.5, 8.0, 6.0, 5.0):
        slope, c_lo, c_hi = fit_exponent(grid, env, l_max)
        print(f"  fit below L = {l_max:4.1f} ({c_lo:8.1e} .. {c_hi:8.1e} PF-days): "
              f"L ~ C^{slope:+.4f}")
    print("  (the paper's own fit to its frontier, eq. 1.3, is C^-0.050)")
    med, lo, hi = above_convergence(data, grid, env, owner)
    print(f"  along the envelope, loss / converged loss of the owning size: "
          f"median {med:.3f}, range {lo:.3f}..{hi:.3f}")

    print("\nwhere each size owns the frontier (the diamonds, on the labelled six):")
    for i in range(len(com)):
        m = owned(grid, env, mat, i)
        if m.sum() < 3:
            print(f"   idx{i:2d} N = {com[i]['n']:9.2e}  owns nothing")
            continue
        c, l = optimum(grid, env, mat, i)
        p = plats[i]
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
