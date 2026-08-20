"""Report figure: the whole scaling-law study in one 2x3 panel."""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from assocmem import fit

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]  # ordinal, light->dark
ACCENT = "#eb6834"
GRID = "#e6e5e1"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK3, "ytick.color": INK3, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "axes.axisbelow": True,
})


def sci(c):
    e = int(np.floor(np.log10(c)))
    m = c / 10**e
    if abs(m - 1) < 0.05:
        return f"$10^{{{e}}}$"
    return f"${m:.1f}".rstrip("0").rstrip(".") + f"\\times10^{{{e}}}$"


def main():
    rows = []
    for nm in ("round1", "round2", "round3"):
        rows += json.loads((RES / f"{nm}.json").read_text())["rows"]
    ff = json.loads((RES / "final_fit.json").read_text())
    hero = json.loads((RES / "hero.json").read_text())
    l_inf = ff["l_inf"]
    cs = [r["c"] for r in ff["rungs"]]
    nstar = np.array([r["n_star"] for r in ff["rungs"]])
    lstar = np.array([r["l_star"] for r in ff["rungs"]])

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    (a, b, c_), (d, e, f) = axes

    # ---- (a) IsoFLOP profiles -------------------------------------------------
    for i, cc in enumerate(cs):
        ns = sorted({r["n"] for r in rows if r["c"] == cc})
        L = [min(r["loss"] for r in rows if r["c"] == cc and r["n"] == n) for n in ns]
        a.plot(ns, L, "o-", color=RAMP[i], ms=4.5, label=sci(cc))
        fo = fit.isoflop_optimum(ns, L)
        xs = np.exp(np.linspace(np.log(min(ns)), np.log(max(ns)), 100))
        a.plot(xs, np.polyval(fo.coef, np.log(xs)), color=RAMP[i], lw=1.0, ls=":", alpha=.8)
        a.plot([fo.n_star], [fo.loss_star], "v", color=RAMP[i], ms=7,
               mec=SURFACE, mew=1.2, zorder=5)
    a.plot([hero["n"]], [hero["loss_exact_screening"]], "*", color=ACCENT, ms=15,
           mec=SURFACE, mew=1.2, zorder=6, label="hero run")
    a.set(xscale="log", xlabel="embedding dim $n$  (params $=512n$)",
          ylabel="test loss  (nats)", ylim=(3.2, 4.92),
          title="a  IsoFLOP profiles ($\\blacktriangledown$ = optimum)")
    a.legend(title="flops", fontsize=7.2, title_fontsize=7.2, ncol=3,
             labelcolor=INK2, loc="upper left", columnspacing=1.1, handletextpad=0.5)

    # ---- (b) n*(C) -----------------------------------------------------------
    an, bn = ff["n_law"]["a"], ff["n_law"]["b"]
    xs = np.array([2e9, 8e12])
    b.plot(xs, an * xs**bn, color=INK3, lw=1.4, ls="--", zorder=1,
           label=f"$n^*=${an:.2g}$\\,C^{{{bn:.3f}}}$")
    b.plot(cs, nstar, "o", color=RAMP[3], ms=7, mec=SURFACE, mew=1.2, label="screening")
    b.plot([hero["c_train"]], [hero["n"]], "*", color=ACCENT, ms=15, mec=SURFACE,
           mew=1.2, label="hero (chosen)")
    b.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
          ylabel="optimal $n^*$", title="b  optimal model size")
    b.legend(fontsize=7.2, labelcolor=INK2, loc="upper left")

    # ---- (c) lr*(C) ----------------------------------------------------------
    al, pl = ff["lr_law"]["a"], ff["lr_law"]["p"]
    c_.plot(xs, al * xs**pl, color=INK3, lw=1.4, ls="--", zorder=1,
            label=f"$\\eta^*=${al:.3g}$\\,C^{{{pl:.3f}}}$")
    c_.plot([4e9, 3e11], [0.0764, 0.0273], "o", color=RAMP[3], ms=7, mec=SURFACE,
            mew=1.2, label="lr sweeps (bracketed)")
    c_.plot([hero["c_train"]], [hero["lr_max"]], "*", color=ACCENT, ms=15, mec=SURFACE,
            mew=1.2, label="hero (chosen)")
    c_.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
           ylabel="optimal peak lr", title="c  optimal learning rate")
    c_.legend(fontsize=7.2, labelcolor=INK2, loc="lower left")

    # ---- (d) L*(C) -----------------------------------------------------------
    aL, bL = ff["l_law"]["a"], ff["l_law"]["b"]
    d.plot(xs, aL * xs**bL, color=INK3, lw=1.4, ls="--", zorder=1,
           label=f"{aL:.3g}$\\,C^{{{bL:.4f}}}$")
    d.plot(cs, lstar - l_inf, "o", color=RAMP[3], ms=7, mec=SURFACE, mew=1.2,
           label="screening $L^*-L_\\infty$")
    d.plot([hero["c_train"]], [hero["predicted_pinned"] - l_inf], "o", ms=11, mfc="none",
           mec=ACCENT, mew=2.0, label="hero predicted")
    d.plot([hero["c_train"]], [hero["loss_exact_screening"] - l_inf], "*", color=ACCENT,
           ms=15, mec=SURFACE, mew=1.2, label="hero actual")
    d.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
          ylabel="excess loss  $L-L_\\infty$  (nats)",
          title=f"d  loss law  ($L_\\infty={l_inf:.3f}$)")
    d.legend(fontsize=7.2, labelcolor=INK2, loc="lower left")

    # ---- (e) hero learning curve ---------------------------------------------
    e.plot(hero["curve_steps"], hero["curve_loss"], "o-", color=ACCENT, ms=5,
           mec=SURFACE, mew=1.0)
    e.axhline(l_inf, color=INK3, lw=1.4, ls="--")
    e.text(hero["steps"] * 0.02, l_inf + 0.03,
           f"irreducible  $L_\\infty={l_inf:.3f}$", color=INK2, fontsize=7.2)
    e.annotate(f"{hero['curve_loss'][-1]:.4f}  (2k-token probe)",
               (hero["curve_steps"][-1], hero["curve_loss"][-1]),
               textcoords="offset points", xytext=(-2, -15), ha="right",
               fontsize=7.5, color=INK2)
    e.text(hero["steps"] * 0.02, 3.63,
           f"final loss, 65k held-out tokens:  {hero['loss_exact_A']:.4f} nats",
           color=INK, fontsize=8.2, fontweight="bold")
    e.set(xlabel="step", ylabel="test loss  (nats)", ylim=(2.35, 3.75),
          title=f"e  hero run  ($n$={hero['n']}, {hero['steps']} steps)")

    # ---- (f) residuals -------------------------------------------------------
    resid = (lstar - l_inf) - aL * np.array(cs) ** bL
    f.axhline(0, color=INK3, lw=1.2)
    f.plot(cs, 1000 * resid, "o", color=RAMP[3], ms=7, mec=SURFACE, mew=1.2,
           label="screening rungs")
    hr = hero["loss_exact_screening"] - hero["predicted_pinned"]
    f.plot([hero["c_train"]], [1000 * hr], "*", color=ACCENT, ms=15, mec=SURFACE,
           mew=1.2, label="hero")
    f.annotate(f"{1000 * hr:+.0f} mnats", (hero["c_train"], 1000 * hr),
               textcoords="offset points", xytext=(-8, -12), ha="right",
               fontsize=7.5, color=ACCENT)
    f.set(xscale="log", xlabel="compute $C$ (flops)", ylabel="actual $-$ fit  (mnats)",
          title="f  fit residuals", ylim=(-30, 40))
    f.legend(fontsize=7.2, labelcolor=INK2, loc="upper left")

    fig.suptitle("Associative memory  $\\hat p(\\cdot|x)=\\mathrm{softmax}(Wx)$,  "
                 "$d=512$  —  scaling-law study under a $10^{13}$ flop budget",
                 fontsize=10.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "scaling_laws.png", dpi=170)
    print("-> figures/scaling_laws.png")

    # table view (accessibility: numbers behind every mark)
    lines = ["| C (flops) | n* | L* | lr used | excess L*-L_inf | fit resid (mnats) |",
             "|---|---|---|---|---|---|"]
    for cc, ns_, ls_, rr in zip(cs, nstar, lstar, resid):
        lines.append(f"| {cc:.3g} | {ns_:.0f} | {ls_:.4f} | {al * cc ** pl:.4f} | "
                     f"{ls_ - l_inf:.4f} | {1000 * rr:+.1f} |")
    lines.append(f"| {hero['c_train']:.4g} (hero) | {hero['n']} | "
                 f"{hero['loss_exact_screening']:.4f} | {hero['lr_max']:.4f} | "
                 f"{hero['loss_exact_screening'] - l_inf:.4f} | {1000 * hr:+.1f} |")
    (RES / "table.md").write_text("\n".join(lines) + "\n")
    print("-> results/table.md")


if __name__ == "__main__":
    main()
