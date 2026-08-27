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

Extraction is per-colour and per-column.  Each curve is one exact RGB, so several
disjoint runs of that colour in a column mean the curve was partly painted over
by a neighbour, not that two curves were caught -- their mean is still on it, so
the mean is what is taken.  (Continuity tracking was tried instead and is worse:
a curve that is fully occluded inside the bundle never re-acquires, and the rest
of its columns are lost.)

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

def loss_of_y(y): return 10.0 - 6.0 * (y - 129.5) / 234.5
def y_of_loss(L): return 129.5 + (10.0 - L) * 234.5 / 6.0
def n_of_barx(p): return 10.0 ** (3.0 + (p - 6.5) / 25.667)

PANELS = {
    "tokens":  dict(x0=132, x1=560, at=lambda x: 10.0**(7.0 + (x-241.0)/70.875),
                    px=lambda t: 241.0 + (np.log10(t)-7.0)*70.875),
    "compute": dict(x0=600, x1=1045, at=lambda x: 10.0**(-9.0 + (x-689.5)/33.433),
                    px=lambda c: 689.5 + (np.log10(c)+9.0)*33.433),
}

def runs_of(mask_col):
    """Contiguous row-runs of a boolean column, as (centre, length)."""
    rows = np.where(mask_col)[0]
    if not len(rows):
        return []
    out, run = [], [rows[0]]
    for r in rows[1:]:
        if r - run[-1] <= 2:
            run.append(r)
        else:
            out.append((float(np.mean(run)), len(run))); run = [r]
    out.append((float(np.mean(run)), len(run)))
    return out

def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGE_URL, SRC)
        print(f"fetched {IMAGE_URL}")
    im = np.asarray(Image.open(SRC).convert("RGB")).astype(int)
    bar = im[156, 1075:1239, :]

    sub = im[TOP:BOT, 132:560].reshape(-1, 3)
    mx, mn = sub.max(1), sub.min(1)
    uniq, cnt = np.unique(sub[(mx - mn) > 45], axis=0, return_counts=True)
    cand = sorted([(int(c), u, int(((bar - u)**2).sum(1).argmin()))
                   for u, c in zip(uniq, cnt) if c >= 150], key=lambda t: -t[0])
    chosen = []
    for c, u, p in cand:
        if all(abs(p - q) >= 7 for _, _, q in chosen):
            chosen.append((c, u, p))
    chosen.sort(key=lambda t: t[2])

    out = {"source": {
        "figure": "Kaplan et al. 2020, arXiv:2001.08361, Figure 2",
        "image": "https://ar5iv.labs.arxiv.org/html/2001.08361/assets/"
                 "EfficiencyIllustration.png",
        "method": "per-colour pixel extraction, continuity-tracked; axes and "
                  "colourbar calibrated from the image's own gridlines, ticks "
                  "and colourbar labels",
        "caveats": ["loss axis is linear in the original; x axes are log",
                    "N is read from the colourbar and carries about +-10%",
                    "in the dense bundle a curve can be shadowed by its "
                    "neighbours, so early columns are sparser"],
    }, "curves": {}}

    for name, P in PANELS.items():
        series = []
        for c, u, p in chosen:
            px = im[TOP:BOT, P["x0"]:P["x1"], :]
            hit = abs(px - u).sum(2) <= 12
            xs, ys = [], []
            for j in range(hit.shape[1]):
                rows = np.where(hit[:, j])[0]
                if not len(rows):
                    continue
                # This colour belongs to exactly one curve, so several runs in a
                # column mean the curve was partly painted over by a neighbour,
                # not that two curves were caught: their mean is still on it.
                xs.append(float(P["at"](P["x0"] + j)))
                ys.append(float(loss_of_y(TOP + rows.mean())))
            if len(xs) >= 25:
                series.append({"n": float(n_of_barx(p)),
                               "rgb": [int(v) for v in u],
                               "x": xs, "loss": ys})
        series.sort(key=lambda s: s["n"])
        out["curves"][name] = series

    OUT.write_text(json.dumps(out))

    # The two panels are independent extractions of the same runs, so a colour's
    # converged loss must come out the same in both.  That is the one strong
    # self-check available without the original data.
    tok = {s["rgb"][0] * 65536 + s["rgb"][1] * 256 + s["rgb"][2]: s
           for s in out["curves"]["tokens"]}
    worst = 0.0
    for s in out["curves"]["compute"]:
        k = s["rgb"][0] * 65536 + s["rgb"][1] * 256 + s["rgb"][2]
        if k in tok:
            d = abs(min(s["loss"]) - min(tok[k]["loss"]))
            worst = max(worst, d)
    print(f"\ncross-panel plateau agreement: worst {worst:.3f} nats")
    if worst > 0.15:
        sys.exit("panels disagree -- calibration or extraction is wrong")

    # Overlay, the only check that matters: red must sit on the real ink.
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
            print(f"   N={s['n']:9.2e}  L {max(s['loss']):5.2f} -> {min(s['loss']):4.2f}"
                  f"  {len(s['x']):3d} cols")

main()
