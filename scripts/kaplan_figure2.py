"""Digitize Figure 2 of Kaplan et al. 2020 from the paper's own figure image.

The paper never released these training curves, and its fitted L(N, S) is a
smooth power law that misses what the figure actually shows -- the flat stretch
at the initialization loss, the S-shaped drop, and the noise -- so a curve drawn
from the law looks like a mock-up of the figure rather than the figure.  This
reads the pixels instead.

Calibration, all measured off the image rather than assumed:

  y      grey gridlines at rows 129.5 / 208 / 286 / 364 carry losses 10 / 8 / 6 / 4.
         The spacing is 78.5 / 78 / 78 px, i.e. the axis is LINEAR in loss.
  x      tick marks under the left axis at 241 / 383 / 524.5 px are 1e7 / 1e9 / 1e11
         tokens; under the right axis 689.5 / 790 / 890 / 990.5 px are 1e-9 .. 1e0
         PF-days.  Both are log, 70.875 and 33.433 px per decade.
  colour each curve is one exact viridis RGB.  Model size comes from matching that
         RGB against the figure's own colourbar (x 1075..1238 of row 156), whose
         1e3 / 1e6 / 1e9 labels sit at bar-x 6.5 / 83.5 / 161.5 -- 25.67 px per
         decade.  So N carries the colourbar's quantization, roughly +-10 %.

Three things are measured, in this order, and the docstrings of `own_pixels`,
`column_samples` and `merge` say exactly what each one does and does not assume.

  1. `own_pixels`  which pixels belong to which curve.  Exact colour only, and a
     pixel goes to the reference colour it is *closest* to rather than to every
     reference within a ball of it.  The tolerance CANNOT be loosened to chase the
     anti-aliased edges: this colourmap is nearly a straight line in RGB, so the
     midpoint of curve i-1 and curve i+1 sits 1.5 to 14 (L1) from curve i's own
     reference -- inside the tolerance.  A pixel where two curves overlap is
     therefore indistinguishable, by colour, from a pixel of the curve between
     them, and a looser tolerance invents that middle curve wherever its two
     neighbours cross.  So occluded stretches are left as gaps here, on purpose.
  2. `column_samples`  where the curve is in each column.  Median, then a window
     around a running median of those medians.  A plain mean over every matching
     row was what this script used to do and is what made the drawn family tangle:
     inside the bundle a curve is painted over in part, the surviving rows are the
     top or the bottom of a steep stroke rather than the middle of it, and the mean
     of them walks a quarter of a nat away from the line and back.
  3. `merge`  the two panels are two independent paintings of the same twenty runs
     -- C = 6ND, so the right panel is the left panel shifted in log x -- with
     different occlusion because the axes have different scales (70.9 vs 33.4 px
     per decade).  Fitting one shift per curve and filling the left panel's gaps
     from the right one closes almost all of them with pixels *measured in the
     other panel*.  The residual of that fit, printed per curve, is the strongest
     self-check this script has: nineteen of the twenty curves agree between the
     panels to under one pixel of the coarser axis, and the twentieth to 2.1.

    uv run python scripts/kaplan_figure2.py   # -> results/kaplan_figure2.json
"""
import json, pathlib, sys, urllib.request
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results/cache/kaplan_figure2.png"
OUT = ROOT / "results/kaplan_figure2.json"
OVERLAY = ROOT / "results/cache/kaplan_figure2_overlay.png"
IMAGE_URL = ("https://ar5iv.labs.arxiv.org/html/2001.08361/assets/"
             "EfficiencyIllustration.png")
TOP, BOT = 95, 441

PF_DAY = 8.64e19          # FLOPs in one petaflop-day, for the C = 6ND cross-check
TOL = 12                  # L1 colour tolerance; see `own_pixels` for why it is tight
CORE = 2                  # L1 tolerance for "unambiguously this stroke's colour"
SAT = 45                  # max(RGB) - min(RGB) above which a pixel is coloured ink
TREND = 9                 # running median of column medians, in columns either side
WINDOW = 6                # rows kept around that trend, before averaging
MIN_PIXELS = 100          # pixels of one exact colour before it is a candidate curve
EDGE = 33                 # a curve must reach within this many columns of the right

def loss_of_y(y): return 10.0 - 6.0 * (y - 129.5) / 234.5
def y_of_loss(L): return 129.5 + (10.0 - L) * 234.5 / 6.0
def n_of_barx(p): return 10.0 ** (3.0 + (p - 6.5) / 25.667)

PANELS = {
    "tokens":  dict(x0=132, x1=560, at=lambda x: 10.0**(7.0 + (x-241.0)/70.875),
                    px=lambda t: 241.0 + (np.log10(t)-7.0)*70.875),
    "compute": dict(x0=600, x1=1045, at=lambda x: 10.0**(-9.0 + (x-689.5)/33.433),
                    px=lambda c: 689.5 + (np.log10(c)+9.0)*33.433),
}


def own_pixels(px, refs, i, tol=None):
    """Boolean mask of the pixels of panel slice `px` that belong to curve `i`.

    Two conditions, and the second one is the fix for the mutual contamination the
    old fixed-ball test had: a pixel counts for curve `i` only if `refs[i]` is the
    *nearest* of all twenty reference colours, not merely within TOL of it.  The
    closest pair of reference colours here (N ~ 6.5e4 and 1.1e5) are 10 apart in L1
    and the ball is 12 wide, so under the old test each of them claimed every pixel
    of the other.

    TOL stays at 12, i.e. at the exact stroke colour plus a pixel of rounding.  It is
    tempting to raise it, because that would recover the occluded stretches -- with
    TOL = 40 the worst-covered curve went from 192 to 292 columns.  It is also wrong:
    viridis is locally straight in RGB, so the midpoint of curve i-1 and curve i+1
    sits between 2.5 and 22.5 from curve i's own reference, under 12 for thirteen of
    the eighteen interior curves.  Where two curves overlap the composite is
    therefore nearer to the reference of the curve *between* them than that curve's
    own anti-aliasing is, and a looser tolerance would invent the middle curve out of
    its neighbours' crossing.  This is not hypothetical: the one candidate colour this
    script rejects, RGB (72, 29, 111), is exactly such a composite, and it survived
    every test but "does it reach the right-hand edge".  Gaps are the honest answer;
    what happens to them is `merge`'s job first and, for whatever survives, the
    drawing script's.
    """
    sat = (px.max(2) - px.min(2)) > SAT
    dist = np.abs(px[:, :, None, :] - refs[None, None, :, :]).sum(3)
    return sat & (dist.argmin(2) == i) & (dist.min(2) <= (TOL if tol is None else tol))


def column_samples(own, core):
    """Curve position per column, as (column indices, row positions).

    `own` is `own_pixels` at TOL, `core` the same at CORE -- the pixels that are the
    stroke colour to within rounding, with no room for a blend.

    Per column the CORE rows are reduced by their MEDIAN, which unlike a mean ignores
    a short stray run; those medians are smoothed by a running median over TREND
    columns either side to give a trend; and the reading is then the mean of the `own`
    rows within WINDOW of that trend.  Columns with nothing inside the window are
    dropped rather than guessed at.

    The point of the window is partial occlusion.  Where the stroke is steep it covers
    twenty rows in one column and its centre is the honest reading; where a neighbour
    is painted over its lower half, the centre of what is left is up to ten rows -- a
    quarter of a nat -- too high, and the error flips sign from column to column as the
    overlap moves.  That is what used to make the drawn curves cross each other inside
    the bundle.

    The point of anchoring the trend on CORE rather than on TOL is that a curve which
    is fully occluded for a stretch has no exact pixels there but often does have loose
    ones, and those loose ones are composites of *other* curves that happen to sit near
    this one's colour.  Following them, the trend walks onto a neighbouring stroke and
    stays there: on the N ~ 4.8e6 run it produced fifty columns a nat below the truth,
    still perfectly smooth, and it was the single largest thing the drawing script's
    ordering projection then had to undo.  Anchored on CORE the same stretch becomes a
    gap, which is what it is.
    """
    rows = [np.where(own[:, j])[0] for j in range(own.shape[1])]
    crows = [np.where(core[:, j])[0] for j in range(core.shape[1])]
    med = np.array([np.median(r) if len(r) else np.nan for r in crows])
    idx = np.where([len(r) > 0 for r in crows])[0]
    cols, vals = [], []
    for j in range(own.shape[1]):
        if not len(rows[j]) or not len(idx):
            continue
        k = np.searchsorted(idx, j)
        t = np.median(med[idx[max(0, k - TREND):k + TREND + 1]])
        sel = rows[j][np.abs(rows[j] - t) <= WINDOW]
        if len(sel):
            cols.append(int(j)); vals.append(float(sel.mean()))
    return np.array(cols), np.array(vals)


CO_SAMPLED = 0.02         # decades: how near a tokens sample must be to compare
SLOPE_MIN = 0.5           # nats/decade below which a column carries no x information
COMPUTE_PPD = 33.433      # px per decade on the compute axis, the coarser of the two


def disagreement(xt, yt, xc, yc, shift):
    """How far apart the two panels' readings of one run are, at a given shift.

    Returns (x offset in decades, loss difference in nats, steep samples, samples),
    or None if the two do not overlap enough to say.

    Two numbers because one of them is misleading on its own.  The loss difference
    is what you would think to measure, but in the drop the curve falls up to eight
    nats per decade, so half a pixel of residual x misalignment shows up as a
    quarter of a nat and the number says "the extractions disagree" when they in
    fact agree to half a pixel.  Dividing by the local slope undoes that and gives
    the honest quantity: how far apart in D the two panels put the same point of the
    same curve.  Columns flatter than SLOPE_MIN are excluded from it -- on a flat
    stretch the loss difference divided by a near-zero slope is meaningless, and
    equally a flat stretch constrains the shift not at all.

    Only compute samples within CO_SAMPLED decades of an actual tokens sample are
    compared, so this is measurement against measurement.  Without that restriction
    a compute sample sitting inside one of the tokens panel's occlusion gaps would
    be compared against the straight chord `np.interp` draws across the gap, and the
    residual would report the chord's error rather than the extraction's -- exactly
    backwards, since filling those gaps with the other panel's pixels is the point
    of the merge.
    """
    lt, lc = np.log10(xt), np.log10(xc)
    xx = lc + shift
    m = ((np.abs(xx[:, None] - lt[None, :]).min(1) <= CO_SAMPLED)
         & (xx >= lt.min()) & (xx <= lt.max()))
    if m.sum() < 15:
        return None
    smooth = np.convolve(np.pad(yt, 3, mode="edge"), np.ones(7) / 7, "valid")
    slope = np.interp(xx[m], lt, np.gradient(smooth, lt))
    r = np.interp(xx[m], lt, yt) - yc[m]
    nats = float(np.sqrt(np.mean(r ** 2)))
    steep = np.abs(slope) > SLOPE_MIN
    if steep.sum() < 10:
        # Only flat stretches overlap (the smallest run, which the figure paints
        # over for all of its drop).  Report the weakest bound the data supports
        # rather than dividing by a slope that is not there.
        return nats / SLOPE_MIN, nats, 0, int(m.sum())
    dec = float(np.sqrt(np.mean((r[steep] / slope[steep]) ** 2)))
    return dec, nats, int(steep.sum()), int(m.sum())


def fit_shift(xt, yt, xc, yc, guess):
    """The log10 shift that maps the compute panel's x onto the tokens panel's.

    C = 6ND, so for one run the right panel of the figure IS the left panel slid
    along x by log10(PF_DAY / 6N) -- that is the `guess`.  It is only a guess
    because N is read off the colourbar to about +-10 % and because the paper's
    compute accounting is not exactly 6ND, so the shift is instead *fitted*:
    a coarse-to-fine scan for the shift that minimises `disagreement`'s x offset.

    Minimising the x offset rather than the loss difference matters.  Most of the
    overlap between the two panels is flat -- the initialization plateau on the left
    and the converged plateau on the right -- and a flat stretch fits any shift
    equally well while still contributing its noise, so a least-squares in loss is
    dominated by the columns that carry no information about the shift.  Weighting
    by 1/slope moves the fit onto the drop, which is the only part that constrains
    it.  It is worth about a pixel: with the plain least-squares in loss, four curves
    were left one to four pixels out of alignment; minimising the x offset, nineteen of
    the twenty come in under one pixel.
    """
    best = (np.inf, guess, np.inf, 0, 0)
    for step, half in ((0.02, 0.6), (0.002, 0.03)):
        centre = best[1]
        for s in np.arange(centre - half, centre + half + 1e-9, step):
            got = disagreement(xt, yt, xc, yc, float(s))
            if got and got[0] < best[0]:
                best = (got[0], float(s), got[1], got[2], got[3])
    return best[1], best[0], best[2], best[3], best[4]


def merge(xt, yt, xc, yc, shift):
    """Fill the tokens panel's gaps with the compute panel's samples, in tokens.

    Both panels draw the same run, so a column occluded in one and measured in the
    other is still a measurement of it.  The compute samples are mapped into D through
    `shift` and kept only where the tokens panel has nothing within CO_SAMPLED decades
    -- it is gap filling, not pooling.

    Pooling both panels everywhere was the obvious thing and it is worse.  Where the
    two panels disagree, pooling interleaves the two readings sample by sample and the
    curve comes out as a sawtooth between them, which on the three worst-aligned runs
    is nearly a nat tall; every one of the big ordering violations left downstream came
    from that.  Preferring the tokens panel wherever it has a reading is also the right
    way round on resolution: its axis is 70.9 px per decade against the compute panel's
    33.4, so a tokens column is a finer measurement, and the drop is steep enough in x
    that resolution in x is what limits the reading.

    Nothing is interpolated here: a gap that is a gap in both panels stays a gap.
    """
    lt = np.log10(xt)
    lc = np.log10(xc) + shift
    if len(lt):
        far = np.abs(lc[:, None] - lt[None, :]).min(1) > CO_SAMPLED
    else:
        far = np.ones(len(lc), bool)
    lx = np.concatenate([lt, lc[far]])
    y = np.concatenate([yt, yc[far]])
    o = np.argsort(lx)
    return 10.0 ** lx[o], y[o]


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGE_URL, SRC)
        print(f"fetched {IMAGE_URL}")
    im = np.asarray(Image.open(SRC).convert("RGB")).astype(int)
    bar = im[156, 1075:1239, :]

    # Which colours are curves.  Every exact colour with at least MIN_PIXELS pixels in
    # the left panel is a candidate, placed on the colourbar to get its N.  This used to
    # dedup candidates that landed within 7 px of each other on the colourbar, i.e.
    # within 0.27 decades of N, and that was wrong: the figure's sizes are spaced 0.1 to
    # 0.6 decades apart, so the rule threw away six real curves and then let their pixels
    # be claimed by whichever of the survivors was nearest in colour.  That is what put
    # two disjoint strokes of "one" colour in a single column, which the old comment in
    # this file explained away as one curve painted over by a neighbour.  The candidate
    # set is stable: 21 candidates for any MIN_PIXELS from 60 to 150.
    #
    # Twenty of the 21 are curves.  The odd one out, RGB (72, 29, 111) at N ~ 1.3e3, is
    # not: it appears only over columns 150..333 and never reaches the right-hand edge of
    # the panel, where every real run is sitting on its converged loss.  It is the
    # composite of the two smallest runs over the stretch where they overlap -- both of
    # them are there in the same columns, under it.  So the test for a curve is that it
    # reaches the last EDGE columns of the panel.
    sub = im[TOP:BOT, 132:560].reshape(-1, 3)
    mx, mn = sub.max(1), sub.min(1)
    uniq, cnt = np.unique(sub[(mx - mn) > SAT], axis=0, return_counts=True)
    cand = sorted([(int(c), u, int(((bar - u)**2).sum(1).argmin()))
                   for u, c in zip(uniq, cnt) if c >= MIN_PIXELS], key=lambda t: -t[0])
    dedup = []
    for c, u, p in cand:
        if all(abs(p - q) >= 2 for _, _, q in dedup):
            dedup.append((c, u, p))
    dedup.sort(key=lambda t: t[2])

    P = PANELS["tokens"]
    px = im[TOP:BOT, P["x0"]:P["x1"], :]
    refs_all = np.array([u for _, u, _ in dedup])
    chosen = []
    for i, (c, u, p) in enumerate(dedup):
        cols = np.where(own_pixels(px, refs_all, i).sum(0) > 0)[0]
        if len(cols) and cols.max() >= (P["x1"] - P["x0"]) - EDGE:
            chosen.append((c, u, p))
        else:
            print(f"   not a curve: rgb {list(u)} (N ~ {n_of_barx(p):.2e}) stops at "
                  f"column {cols.max() if len(cols) else -1} of "
                  f"{P['x1'] - P['x0'] - 1}; it is where two runs overlap, not a run")
    refs = np.array([u for _, u, _ in chosen])
    print(f"   {len(chosen)} curves from {len(dedup)} candidate colours\n")

    # Per panel, per curve: the raw per-column reading, before the panels are merged.
    raw = {}
    for name, P in PANELS.items():
        px = im[TOP:BOT, P["x0"]:P["x1"], :]
        rows = []
        for i in range(len(chosen)):
            cols, vals = column_samples(own_pixels(px, refs, i),
                                        own_pixels(px, refs, i, CORE))
            rows.append((np.array([P["at"](P["x0"] + j) for j in cols]),
                         np.array([loss_of_y(TOP + v) for v in vals])))
        raw[name] = rows

    out = {"source": {
        "figure": "Kaplan et al. 2020, arXiv:2001.08361, Figure 2",
        "image": "https://ar5iv.labs.arxiv.org/html/2001.08361/assets/"
                 "EfficiencyIllustration.png",
        "method": "per-colour pixel extraction at the exact stroke colour, nearest "
                  "reference wins; per column a median and a window around a "
                  "running median of those medians; the two panels, which draw the "
                  "same runs at different x scales, merged per curve on a fitted "
                  "log-x shift.  Axes and colourbar calibrated from the image's own "
                  "gridlines, ticks and colourbar labels",
        "caveats": ["loss axis is linear in the original; x axes are log",
                    "N is read from the colourbar and carries about +-10%",
                    "the colour tolerance cannot be loosened without inventing a "
                    "curve out of the overlap of its two neighbours, so a stretch "
                    "where a curve is fully painted over is left as a gap here",
                    "each panel's x for a curve is the merged D mapped back through "
                    "that curve's fitted shift, so the two panels are exactly the "
                    "same samples; the per-curve fit residual is reported instead "
                    "of the old cross-panel plateau agreement"],
    }, "curves": {}, "merge": []}

    worst = 0.0
    tok, com = [], []
    print("panel merge, per curve.  C = 6ND, so the compute panel is the tokens panel"
          "\nslid along log x; one shift is fitted per curve and the tokens panel's gaps"
          "\nare filled from the compute panel."
          "\n'apart' is how far the two panels put the same point of the same curve,"
          "\nin decades of D and in px of the coarser (compute) axis.\n")
    for i, (c, u, p) in enumerate(chosen):
        n = n_of_barx(p)
        xt, yt = raw["tokens"][i]
        xc, yc = raw["compute"][i]
        guess = float(np.log10(PF_DAY / (6.0 * n)))
        shift, dec, nats, steep, npts = fit_shift(xt, yt, xc, yc, guess)
        xm, ym = merge(xt, yt, xc, yc, shift)
        worst = max(worst, dec)
        print(f"   N={n:9.2e}  shift {shift:6.3f} (6ND predicts {guess:6.3f}, "
              f"d={shift - guess:+.3f})  apart {dec:.3f} dec = "
              f"{dec * COMPUTE_PPD:4.2f} px ({nats:.3f} nats) on {steep:3d}/{npts:3d}"
              f" pts  {len(xt):3d}+{len(xc):3d} -> {len(xm):3d} cols")
        out["merge"].append({"n": float(n), "shift": shift, "apart_decades": dec,
                             "apart_nats": nats, "steep": steep, "overlap": npts,
                             "shift_6nd": guess})
        rec = dict(n=float(n), rgb=[int(v) for v in u])
        tok.append(dict(rec, x=[float(v) for v in xm],
                        loss=[float(v) for v in ym]))
        com.append(dict(rec, x=[float(v) for v in xm / 10.0 ** shift],
                        loss=[float(v) for v in ym]))
    out["curves"]["tokens"] = tok
    out["curves"]["compute"] = com
    OUT.write_text(json.dumps(out))

    # The self-check.  The two panels are independent extractions of the same runs at
    # different pixel scales, so the residual of the one-parameter fit that aligns
    # them bounds the error of both -- a far stronger statement than the converged
    # loss agreeing, which this script used to check, since it holds along the whole
    # curve rather than at one point.  Nineteen of the twenty come in under one pixel;
    # the one that does not (N ~ 6.5e4) disagrees only over the upper knee of its drop,
    # where the curve falls eight nats per decade and a pixel of x is a quarter of a
    # nat.  3 px is the bound that lets that through while still failing on a
    # mis-calibrated axis, which would be wrong everywhere at once.
    print(f"\ncross-panel agreement after the fitted shift: worst "
          f"{worst:.3f} decades = {worst * COMPUTE_PPD:.2f} compute-axis px")
    if worst * COMPUTE_PPD > 3.0:
        sys.exit("panels disagree -- calibration or extraction is wrong")

    # Overlay, the only check that matters: red must sit on the real ink.  Now that
    # the panels are merged, most of the red in one panel comes from pixels measured
    # in the *other*, so the overlay also checks the merge: if a shift were wrong the
    # imported samples would land beside the stroke instead of on it.
    a = np.asarray(Image.open(SRC).convert("RGB")).copy()
    for name, P in PANELS.items():
        for s in out["curves"][name]:
            for xv, lv in zip(s["x"], s["loss"]):
                xi, yi = int(round(P["px"](xv))), int(round(y_of_loss(lv)))
                if 0 <= yi < a.shape[0] and 0 <= xi < a.shape[1]:
                    a[yi, xi] = [255, 0, 0]
    Image.fromarray(a).save(OVERLAY)

    for name in PANELS:
        ss = out["curves"][name]
        print(f"{name:8s} {len(ss)} curves")
        for s in ss:
            lx = np.log10(s["x"])
            print(f"   N={s['n']:9.2e}  L {max(s['loss']):5.2f} -> {min(s['loss']):4.2f}"
                  f"  {len(s['x']):3d} cols  x {s['x'][0]:8.2e}..{s['x'][-1]:8.2e}"
                  f"  widest gap {np.diff(lx).max():.2f} dec")

main()
