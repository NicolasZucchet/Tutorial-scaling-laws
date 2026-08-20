"""Final report figure: the best run's laws, plus the compute-allocation study."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from assocmem import fit
from assocmem.plots import ACCENT, GRID, INK, INK2, INK3, RAMP, STYLE, SURFACE, _ramp, sci

ROOT = pathlib.Path(__file__).resolve().parents[1]
L_INF = 2.4632  # 65536-token eval set

ATTEMPTS = [  # (label, C_train, hero loss, screening flops, tuning share)
    ("careful\n47% tuning", 5.009e12, 3.2993, 4.713e12),
    ("defaults\n26% tuning", 6.971e12, 3.2765, 2.640e12),
    ("expert\n8% tuning", 8.826e12, 3.2609, 7.853e11),
]


def main():
    st = json.loads((ROOT / "runs/expert/state.json").read_text())
    hero, rows = st["hero"], st["rows"]
    ex = json.loads((ROOT / "runs/expert/expert_run.json").read_text())
    laws = ex["laws"]
    an, bn, r2n = laws["n_law"]
    a_lr, p_lr = laws["lr_law"]
    aL, alL, r2L = laws["loss_law"]
    cs = [r["c"] for r in laws["rungs"]]
    hc = hero["c_train"]

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5))
        (a, b, c_), (d, e, f) = axes

        # (a) IsoFLOP profiles ------------------------------------------------
        rungs = sorted({r["c"] for r in rows})
        for i, cc in enumerate(rungs):
            sub = sorted((r for r in rows if r["c"] == cc), key=lambda r: r["n"])
            ns = sorted({r["n"] for r in sub})
            loss = [min(r["loss"] for r in sub if r["n"] == n) for n in ns]
            col = _ramp(i, len(rungs))
            a.plot(ns, loss, "o-", ms=4.5, color=col, label=sci(cc))
            fo = fit.isoflop_optimum(ns, loss)
            xs = np.exp(np.linspace(np.log(min(ns)), np.log(max(ns)), 100))
            a.plot(xs, np.polyval(fo.coef, np.log(xs)), color=col, lw=1.0, ls=":", alpha=.8)
            a.plot([fo.n_star], [fo.loss_star], "v", color=col, ms=7, mec=SURFACE,
                   mew=1.2, zorder=5)
        a.plot([hero["n"]], [hero["loss"]], "*", color=ACCENT, ms=15, mec=SURFACE,
               mew=1.2, zorder=6, label="hero")
        a.set(xscale="log", xlabel="width $n$  (params $=512n$)",
              ylabel="test loss (nats)", ylim=(3.15, 4.95),
              title="a  IsoFLOP profiles ($\\blacktriangledown$ = optimum)")
        a.legend(title="flops", fontsize=7, title_fontsize=7, labelcolor=INK2, ncol=3,
                 loc="upper left", columnspacing=1.0, handletextpad=0.5)

        # (b) n*(C) -----------------------------------------------------------
        xs = np.array([1e9, 2e13])
        b.plot(xs, an * xs**bn, color=INK3, lw=1.4, ls="--",
               label=f"${an:.3g}\\,C^{{{bn:.3f}}}$  $r^2$={r2n:.4f}")
        b.plot(cs, [r["n_star"] for r in laws["rungs"]], "o", color=RAMP[3], ms=7,
               mec=SURFACE, mew=1.2, label="screening")
        b.plot([hc], [hero["n"]], "*", color=ACCENT, ms=15, mec=SURFACE, mew=1.2,
               label="hero (chosen)")
        b.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
              ylabel="optimal width $n^*$", title="b  optimal model size")
        b.legend(fontsize=7, labelcolor=INK2, loc="upper left")

        # (c) lr*(C) ----------------------------------------------------------
        anchors = sorted(_lr_anchors(rows, laws))
        c_.plot(xs, a_lr * xs**p_lr, color=INK3, lw=1.4, ls="--",
                label=f"${a_lr:.3g}\\,C^{{{p_lr:.3f}}}$")
        c_.plot([k for k, _ in anchors], [v for _, v in anchors], "o", color=RAMP[3],
                ms=7, mec=SURFACE, mew=1.2, label="4 lr parabolas")
        c_.plot([hc], [hero["lr_max"]], "*", color=ACCENT, ms=15, mec=SURFACE, mew=1.2,
                label="hero (chosen)")
        c_.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
               ylabel="optimal peak lr", title="c  optimal learning rate")
        c_.legend(fontsize=7, labelcolor=INK2, loc="lower left")

        # (d) which fit extrapolates? -----------------------------------------
        li, aF, alF = laws_free = ex["laws"].get("loss_law_free", (2.7755, 21.58, 0.1276))
        xs2 = np.exp(np.linspace(np.log(1e9), np.log(2e13), 200))
        d.plot(xs2, aL * xs2**-alL, color=INK3, lw=1.4, ls="--",
               label=f"pinned $L_\\infty$: $C^{{-{alL:.4f}}}$")
        d.plot(xs2, li + aF * xs2**-alF - L_INF, color=ACCENT, lw=1.4, ls=":",
               label=f"free $L_\\infty$ (3-param)")
        d.plot(cs, [r["loss_star"] - laws["rungs"][0]["loss_star"] * 0 - 2.4678
                    for r in laws["rungs"]], "o", color=RAMP[3], ms=7, mec=SURFACE,
               mew=1.2, label="screening rungs")
        d.plot([x[1] for x in ATTEMPTS], [x[2] - L_INF for x in ATTEMPTS], "*",
               color=ACCENT, ms=13, mec=SURFACE, mew=1.2, label="3 hero runs (actual)")
        d.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
              ylabel="excess loss $L-L_\\infty$ (nats)",
              title="d  the pinned fit over-predicts at 90$\\times$")
        d.legend(fontsize=6.8, labelcolor=INK2, loc="lower left")

        # (e) the allocation study -------------------------------------------
        cc = np.array([x[1] for x in ATTEMPTS])
        ll = np.array([x[2] for x in ATTEMPTS])
        pb, pa = np.polyfit(np.log(cc), np.log(ll - L_INF), 1)
        xs3 = np.exp(np.linspace(np.log(4e12), np.log(1.05e13), 100))
        e.plot(xs3, L_INF + np.exp(pa) * xs3**pb, color=INK3, lw=1.4, ls="--",
               label=f"$L_\\infty + {np.exp(pa):.2f}\\,C^{{{pb:.4f}}}$  (fit to the 3 runs)")
        e.axvline(9.61e12, color=GRID, lw=9, zorder=0)
        e.text(9.61e12, L_INF + np.exp(pa) * 9.61e12**pb, "zero-tuning ceiling 3.2552",
               fontsize=7, color=INK2, va="bottom", ha="center", rotation=90)
        for i, (lab, c, l, _) in enumerate(ATTEMPTS):
            e.plot([c], [l], "*", color=ACCENT, ms=15, mec=SURFACE, mew=1.2, zorder=5)
            e.annotate(f"{lab}\n{l:.4f}", (c, l), textcoords="offset points",
                       xytext=(8, 6) if i < 2 else (-4, 8), ha="left" if i < 2 else "right",
                       fontsize=7, color=INK)
        e.set(xscale="log", xlim=(4e12, 1.15e13),
              xlabel="compute given to the hero run (flops)",
              ylabel="hero loss (nats)",
              title="e  every flop spent tuning is a flop lost")
        e.legend(fontsize=7, labelcolor=INK2, loc="lower left")

        # (f) learning curve --------------------------------------------------
        f.plot(hero["curve_steps"], hero["curve_loss"], "o-", color=ACCENT, ms=5,
               mec=SURFACE, mew=1.0)
        f.axhline(L_INF, color=INK3, lw=1.4, ls="--")
        f.text(hero["steps"] * 0.04, L_INF + 0.04,
               f"irreducible $L_\\infty$={L_INF:.4f}", color=INK2, fontsize=7.5)
        f.text(hero["steps"] * 0.04, 3.62,
               f"final, 65k held-out tokens: {hero['loss']:.4f} nats", color=INK,
               fontsize=8.5, fontweight="bold")
        f.set(xlabel="step", ylabel="test loss (nats)", ylim=(2.35, 3.75),
              title=f"f  hero run: $n$={hero['n']}, {hero['steps']} steps, "
                    f"lr={hero['lr_max']:.5f}")

        fig.suptitle("Associative memory $\\hat p(\\cdot|x)=\\mathrm{softmax}(Wx)$, $d$=512 — "
                     "best run: 3.2609 nats on $10^{13}$ flops", fontsize=10.5, y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        (ROOT / "figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / "figures/final.png", dpi=170)
        plt.close(fig)
    print("-> figures/final.png")


def _lr_anchors(rows, laws):
    """Recover the per-rung lr parabola vertices for plotting."""
    out = []
    for r in laws["rungs"]:
        cell = sorted((x for x in rows if x["c"] == r["c"]
                       and len({y["lr"] for y in rows if y["c"] == r["c"]
                                and y["n"] == x["n"]}) >= 3), key=lambda x: x["lr"])
        if len(cell) < 3:
            continue
        co = np.polyfit(np.log([x["lr"] for x in cell]), [x["loss"] for x in cell], 2)
        if co[0] > 0:
            out.append((r["c"], float(np.exp(-co[1] / (2 * co[0])))))
    return out


if __name__ == "__main__":
    main()
