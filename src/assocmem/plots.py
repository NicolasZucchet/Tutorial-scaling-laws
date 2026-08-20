"""Plots for the tutorial.  Each round plots itself; the laws plot themselves.

Panel selection is automatic: a sweep with several lrs per cell gets an lr panel, a
sweep with >=3 widths per compute rung gets an IsoFLOP panel, and >=2 rungs get the
emerging n*(C) / L*(C) laws.  Colours are an ordinal single-hue ramp (rungs are
*ordered*, not categorical) plus one accent, validated for CVD separation.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import fit as _fit

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
ACCENT = "#eb6834"
GRID = "#e6e5e1"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "xtick.color": INK3, "ytick.color": INK3, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 110,
}


def _ramp(i: int, n: int) -> str:
    """i-th of n ordered colours, spread across the ramp (never lighter than step 250)."""
    if n <= 1:
        return RAMP[3]
    return RAMP[int(round(i * (len(RAMP) - 1) / (n - 1)))]


def sci(c: float) -> str:
    e = int(np.floor(np.log10(c)))
    m = c / 10**e
    if abs(m - 1) < 0.05:
        return f"$10^{{{e}}}$"
    return f"${m:.1f}".rstrip("0").rstrip(".") + f"\\times10^{{{e}}}$"


def _interactive() -> bool:
    """True inside Jupyter/IPython, where showing a figure is the point."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def _headroom(ax, frac=0.16):
    """Make space at the top for a legend without letting it sit on the data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def _finish(fig, path, show, tight=True):
    if show is None:
        show = _interactive()
    for ax in fig.get_axes():  # log minor tick labels collide; decades are enough
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "log":
                axis.set_minor_formatter(mpl.ticker.NullFormatter())
    if tight:
        fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def panel_lr(res, ax):
    """loss vs lr, one line per (C, n) cell that has >= 2 lrs."""
    cells = sorted({(r["c"], r["n"]) for r in res.rows})
    cells = [(c, n) for c, n in cells
             if len({r["lr"] for r in res.rows if r["c"] == c and r["n"] == n}) >= 2]
    if not cells:
        ax.axis("off")
        ax.text(.5, .5, "no lr sweep yet\n(vary lr at a fixed (C, n))", ha="center",
                va="center", color=INK2, fontsize=8)
        return
    rungs = sorted({c for c, _ in cells})
    for c, n in cells:
        sub = sorted((r for r in res.rows if r["c"] == c and r["n"] == n),
                     key=lambda r: r["lr"])
        col = _ramp(rungs.index(c), len(rungs))
        ax.plot([r["lr"] for r in sub], [r["loss"] for r in sub], "o-", ms=4.5,
                color=col, label=f"C={sci(c)}, n={n}")
        if len(sub) >= 3:
            i = int(np.argmin([r["loss"] for r in sub]))
            if 0 < i < len(sub) - 1:
                x = np.log([r["lr"] for r in sub])
                co = np.polyfit(x, [r["loss"] for r in sub], 2)
                if co[0] > 0:
                    xs = float(np.exp(-co[1] / (2 * co[0])))
                    ax.plot([xs], [np.polyval(co, np.log(xs))], "v", color=col, ms=7,
                            mec=SURFACE, mew=1.2, zorder=5)
    ax.set(xscale="log", xlabel="peak learning rate", ylabel="test loss (nats)",
           title="learning rate ($\\blacktriangledown$ = parabola optimum)")
    if len(cells) <= 6:
        _headroom(ax)
        ax.legend(fontsize=7, labelcolor=INK2, loc="upper center", ncol=2)


def panel_isoflop(res, ax, hero=None):
    """loss vs n at fixed compute, with the fitted parabola and its minimum."""
    iso = res.isoflop()
    rungs = [d["c"] for d in iso]
    for i, d in enumerate(iso):
        col = _ramp(i, len(iso))
        ax.plot(d["ns"], d["loss"], "o-", ms=4.5, color=col, label=sci(d["c"]))
        f = _fit.isoflop_optimum(d["ns"], d["loss"])
        xs = np.exp(np.linspace(np.log(min(d["ns"])), np.log(max(d["ns"])), 100))
        ax.plot(xs, np.polyval(f.coef, np.log(xs)), color=col, lw=1.0, ls=":", alpha=.8)
        ax.plot([d["n_star"]], [d["loss_star"]], "v", color=col, ms=7, mec=SURFACE,
                mew=1.2, zorder=5)
    for c in sorted({r["c"] for r in res.rows}):
        if c not in rungs:  # rung with <3 widths: show the points, no parabola
            sub = sorted((r for r in res.rows if r["c"] == c), key=lambda r: r["n"])
            ax.plot([r["n"] for r in sub], [r["loss"] for r in sub], "o", ms=4.5,
                    color=INK3, alpha=.7)
    if hero:
        ax.plot([hero["n"]], [hero["loss"]], "*", color=ACCENT, ms=15, mec=SURFACE,
                mew=1.2, zorder=6, label="hero")
    ax.set(xscale="log", xlabel="width $n$   (params $=512n$)",
           ylabel="test loss (nats)",
           title="IsoFLOP profiles ($\\blacktriangledown$ = optimum)")
    _headroom(ax, 0.34)
    ax.legend(title="flops", fontsize=7, title_fontsize=7, labelcolor=INK2, ncol=3,
              loc="upper left", columnspacing=1.0, handletextpad=0.5)


def panel_law(ax, cs, ys, a, b, *, ylabel, title, hero=None, r2=None, logy=True):
    xs = np.array([min(cs) / 2.5, max(cs) * (25 if hero else 2.5)])
    lbl = f"${a:.3g}\\,C^{{{b:.3f}}}$" + (f"   $r^2$={r2:.3f}" if r2 is not None else "")
    ax.plot(xs, a * xs**b, color=INK3, lw=1.4, ls="--", zorder=1, label=lbl)
    ax.plot(cs, ys, "o", color=RAMP[3], ms=7, mec=SURFACE, mew=1.2, label="measured")
    if hero is not None:
        ax.plot([hero[0]], [hero[1]], "*", color=ACCENT, ms=15, mec=SURFACE, mew=1.2,
                label="hero")
    ax.set(xscale="log", yscale="log" if logy else "linear",
           xlabel="compute $C$ (flops)", ylabel=ylabel, title=title)
    ax.legend(fontsize=7, labelcolor=INK2)


def panel_budget(lab, ax):
    """A stat strip: what has been spent, on what, and what is left."""
    ax.axis("off")
    spent, total = lab.spent, lab.budget
    ax.barh([0], [total], color=GRID, height=0.55)
    x = 0.0
    for i, r in enumerate(lab.round_log):
        ax.barh([0], [r["flops"]], left=[x], color=_ramp(i, max(len(lab.round_log), 2)),
                height=0.55, edgecolor=SURFACE, lw=1.5,
                hatch="///" if r["smoke"] else None)
        x += r["flops"]
    if lab.hero_record:
        ax.barh([0], [lab.hero_record["c_train"] + lab.hero_record["c_eval"]], left=[x],
                color=ACCENT, height=0.55, edgecolor=SURFACE, lw=1.5)
    ax.set_xlim(0, total)
    ax.set_ylim(-1.4, 1.1)
    ax.text(0, 0.75, f"{spent:.3g} of {total:.2g} flops spent "
                     f"({100 * spent / total:.1f}%)   |   rounds "
                     f"{lab.rounds_used}/{lab.max_rounds}", fontsize=8.5,
            color=INK, fontweight="bold", va="bottom")
    names = [r["name"] for r in lab.round_log if not r["smoke"]]
    if lab.hero_record:
        names.append("hero")
    ax.text(0, -0.9, "  |  ".join(names), fontsize=7.5, color=INK2, va="top")


# --------------------------------------------------------------------------- #
# composites
# --------------------------------------------------------------------------- #
def plot_round(res, path=None, show=None, lab=None):
    """Auto-composed figure for one round."""
    n_lr = max((len({r["lr"] for r in res.rows if r["c"] == c and r["n"] == n})
                for c, n in {(r["c"], r["n"]) for r in res.rows}), default=1)
    iso = res.isoflop()
    panels = []
    if n_lr >= 2:
        panels.append("lr")
    if iso:
        panels.append("isoflop")
    if len(iso) >= 2:
        panels += ["nstar", "loss"]
    if not panels:
        panels = ["scatter"]

    with mpl.rc_context(STYLE):
        w = 3.6 * len(panels)
        fig, axes = plt.subplots(1, len(panels), figsize=(w, 3.4), squeeze=False)
        axes = list(axes[0])
        for name, ax in zip(panels, axes):
            if name == "lr":
                panel_lr(res, ax)
            elif name == "isoflop":
                panel_isoflop(res, ax)
            elif name == "scatter":
                ax.plot([r["n"] for r in res.rows], [r["loss"] for r in res.rows], "o",
                        color=RAMP[3], ms=6, mec=SURFACE, mew=1.2)
                ax.set(xscale="log", xlabel="width $n$", ylabel="test loss (nats)",
                       title="runs this round")
            elif name == "nstar":
                cs = [d["c"] for d in iso]
                a, b, r2 = _fit.powerlaw(cs, [d["n_star"] for d in iso])
                panel_law(ax, cs, [d["n_star"] for d in iso], a, b,
                          ylabel="optimal width $n^*$", title="optimal size so far", r2=r2)
            elif name == "loss":
                cs = [d["c"] for d in iso]
                ax.plot(cs, [d["loss_star"] for d in iso], "o-", color=RAMP[3], ms=7,
                        mec=SURFACE, mew=1.2)
                ax.set(xscale="log", xlabel="compute $C$ (flops)",
                       ylabel="best loss $L^*$ (nats)", title="best loss so far")
        fig.suptitle(f"round: {res.name}", fontsize=10, y=1.04)
        return _finish(fig, path, show)


def plot_laws(laws, path=None, show=None, hero=None):
    """The three fitted laws, with the hero extrapolation if it has been run."""
    cs = [r["c"] for r in laws.rungs]
    an, bn, r2n = laws.n_law
    a_lr, p_lr = laws.lr_law
    aL, alL, r2L = laws.loss_law
    hc = hero["c_train"] if hero else None
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        panel_law(axes[0], cs, [r["n_star"] for r in laws.rungs], an, bn,
                  ylabel="optimal width $n^*$", title="a  optimal model size", r2=r2n,
                  hero=(hc, hero["n"]) if hero else None)
        if laws.lr_anchors:
            panel_law(axes[1], [c for c, _ in laws.lr_anchors],
                      [v for _, v in laws.lr_anchors], a_lr, p_lr,
                      ylabel="optimal peak lr", title="b  optimal learning rate",
                      hero=(hc, hero["lr_max"]) if hero else None)
        else:
            axes[1].axis("off")
            axes[1].text(.5, .5, "no bracketed lr sweep yet", ha="center", color=INK2)
        ax = axes[2]
        xs = np.array([min(cs) / 2.5, max(cs) * (25 if hero else 2.5)])
        ax.plot(xs, aL * xs**-alL, color=INK3, lw=1.4, ls="--",
                label=f"${aL:.3g}\\,C^{{-{alL:.4f}}}$   $r^2$={r2L:.4f}")
        ax.plot(cs, [r["loss_star"] - laws.l_inf for r in laws.rungs], "o", color=RAMP[3],
                ms=7, mec=SURFACE, mew=1.2, label="measured")
        if hero:
            ax.plot([hc], [hero["predicted"] - laws.l_inf], "o", ms=11, mfc="none",
                    mec=ACCENT, mew=2.0, label="hero predicted")
            ax.plot([hc], [hero["loss"] - laws.l_inf], "*", color=ACCENT, ms=15,
                    mec=SURFACE, mew=1.2, label="hero actual")
        ax.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
               ylabel="excess loss $L-L_\\infty$ (nats)",
               title=f"c  loss law  ($L_\\infty$={laws.l_inf:.3f})")
        ax.legend(fontsize=7, labelcolor=INK2)
        return _finish(fig, path, show)


def plot_hero(rec, laws=None, path=None, show=None):
    """Learning curve + where the hero landed relative to the law."""
    with mpl.rc_context(STYLE):
        ncol = 2 if laws else 1
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol + 2, 3.4), squeeze=False)
        ax = axes[0][0]
        ax.plot(rec["curve_steps"], rec["curve_loss"], "o-", color=ACCENT, ms=5,
                mec=SURFACE, mew=1.0)
        ax.axhline(rec["irreducible"], color=INK3, lw=1.4, ls="--")
        ax.text(rec["steps"] * 0.02, rec["irreducible"] + 0.03,
                f"irreducible $L_\\infty$={rec['irreducible']:.3f}", color=INK2, fontsize=7.5)
        ax.axhline(rec["predicted"], color=INK3, lw=1.2, ls=":")
        ax.text(rec["steps"] * 0.02, rec["predicted"] + 0.03,
                f"predicted {rec['predicted']:.4f}", color=INK2, fontsize=7.5)
        ax.set(xlabel="step", ylabel="test loss (nats)",
               title=f"hero run: n={rec['n']}, {rec['steps']} steps, "
                     f"lr={rec['lr_max']:.4f}")
        ax.annotate(f"actual {rec['loss']:.4f}",
                    (rec["curve_steps"][-1], rec["curve_loss"][-1]),
                    textcoords="offset points", xytext=(-4, -14), ha="right",
                    fontsize=8.5, color=INK, fontweight="bold")
        if laws:
            cs = [r["c"] for r in laws.rungs]
            aL, alL, r2L = laws.loss_law
            ax2 = axes[0][1]
            xs = np.array([min(cs) / 2.5, rec["c_train"] * 2.5])
            ax2.plot(xs, aL * xs**-alL, color=INK3, lw=1.4, ls="--", label="fitted law")
            ax2.plot(cs, [r["loss_star"] - laws.l_inf for r in laws.rungs], "o",
                     color=RAMP[3], ms=7, mec=SURFACE, mew=1.2, label="screening")
            ax2.plot([rec["c_train"]], [rec["predicted"] - laws.l_inf], "o", ms=11,
                     mfc="none", mec=ACCENT, mew=2.0, label="predicted")
            ax2.plot([rec["c_train"]], [rec["loss"] - laws.l_inf], "*", color=ACCENT,
                     ms=15, mec=SURFACE, mew=1.2, label="actual")
            ax2.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
                    ylabel="excess loss $L-L_\\infty$ (nats)",
                    title=f"extrapolation: error {rec['loss'] - rec['predicted']:+.4f} nats")
            ax2.legend(fontsize=7, labelcolor=INK2)
        return _finish(fig, path, show)


def plot_summary(lab, laws=None, path=None, show=None):
    """Budget strip + everything measured so far.  The 'where am I' plot."""
    res = lab.results
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 6.2))
        gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 3, 3], hspace=0.55, wspace=0.28)
        panel_budget(lab, fig.add_subplot(gs[0, :]))
        panel_isoflop(res, fig.add_subplot(gs[1, 0]), hero=lab.hero_record)
        panel_lr(res, fig.add_subplot(gs[1, 1]))
        ax_curve = fig.add_subplot(gs[1, 2])
        if lab.hero_record:
            r = lab.hero_record
            ax_curve.plot(r["curve_steps"], r["curve_loss"], "o-", color=ACCENT, ms=5)
            ax_curve.axhline(r["irreducible"], color=INK3, lw=1.4, ls="--")
            ax_curve.text(r["steps"] * 0.16, r["irreducible"] + 0.03,
                          f"irreducible {r['irreducible']:.3f}", color=INK2, fontsize=7.5)
            ax_curve.set(xlabel="step", ylabel="test loss (nats)",
                         title=f"c  hero run: {r['loss']:.4f} nats "
                               f"(predicted {r['predicted']:.4f})")
        else:
            ax_curve.axis("off")
            ax_curve.text(.5, .5, "hero run not done yet", ha="center", color=INK2)
        if laws is not None:
            cs = [r["c"] for r in laws.rungs]
            an, bn, r2n = laws.n_law
            hc = lab.hero_record["c_train"] if lab.hero_record else None
            panel_law(fig.add_subplot(gs[2, 0]), cs, [r["n_star"] for r in laws.rungs],
                      an, bn, ylabel="optimal $n^*$", title="d  optimal size", r2=r2n,
                      hero=(hc, lab.hero_record["n"]) if lab.hero_record else None)
            if laws.lr_anchors:
                a_lr, p_lr = laws.lr_law
                panel_law(fig.add_subplot(gs[2, 1]), [c for c, _ in laws.lr_anchors],
                          [v for _, v in laws.lr_anchors], a_lr, p_lr,
                          ylabel="optimal peak lr", title="e  optimal lr",
                          hero=(hc, lab.hero_record["lr_max"]) if lab.hero_record else None)
            aL, alL, r2L = laws.loss_law
            ax = fig.add_subplot(gs[2, 2])
            xs = np.array([min(cs) / 2.5, (hc or max(cs)) * 2.5])
            ax.plot(xs, aL * xs**-alL, color=INK3, lw=1.4, ls="--",
                    label=f"${aL:.3g}\\,C^{{-{alL:.4f}}}$")
            ax.plot(cs, [r["loss_star"] - laws.l_inf for r in laws.rungs], "o",
                    color=RAMP[3], ms=7, mec=SURFACE, mew=1.2, label="measured")
            if lab.hero_record:
                ax.plot([hc], [lab.hero_record["loss"] - laws.l_inf], "*", color=ACCENT,
                        ms=15, mec=SURFACE, mew=1.2, label="hero")
            ax.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
                   ylabel="excess loss (nats)", title="f  loss law")
            ax.legend(fontsize=7, labelcolor=INK2)
        fig.suptitle(f"lab '{lab.name}'", fontsize=11, y=0.99)
        return _finish(fig, path, show, tight=False)
