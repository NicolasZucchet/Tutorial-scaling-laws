"""Plots for the tutorial.  Each round plots itself; the fits are the student's to make.

Panel selection is automatic: a sweep with several lrs per cell gets an lr panel, a
sweep with >=3 model sizes per compute rung gets an IsoFLOP panel, and >=2 rungs get the
measured n*(C).  `plot_runs` is the general one -- any column against any other, with a
third on the colour axis and your own fit overlaid.  Colours are an ordinal single-hue
ramp (rungs are *ordered*, not categorical) plus one accent, validated for CVD
separation.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import fit as _fit
from .data import D_OUT
from .train import BATCH

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
        if c not in rungs:  # rung with <3 sizes: show the points, no parabola
            sub = sorted((r for r in res.rows if r["c"] == c), key=lambda r: r["n"])
            ax.plot([r["n"] for r in sub], [r["loss"] for r in sub], "o", ms=4.5,
                    color=INK3, alpha=.7)
    if hero:
        ax.plot([hero["n"]], [hero["loss"]], "*", color=ACCENT, ms=15, mec=SURFACE,
                mew=1.2, zorder=6, label="hero")
    ax.set(xscale="log", xlabel="parameters $N$",
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
                height=0.55, edgecolor=SURFACE, lw=1.5)
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
    names = [r["name"] for r in lab.round_log]
    if lab.hero_record:
        names.append("hero")
    ax.text(0, -0.9, "  |  ".join(names), fontsize=7.5, color=INK2, va="top")


# --------------------------------------------------------------------------- #
# composites
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# the generic explorer: any column against any column, coloured by a third
# --------------------------------------------------------------------------- #
COLUMNS = ("c", "n", "d", "tokens", "lr", "loss", "excess", "params", "width",
           "tokens_per_param", "flops", "round", "seed", "init", "steps",
           "n_star", "loss_star")

_AXIS_LABEL = {
    "c": "compute $C$ (flops)", "n": "parameters $N$", "steps": "steps",
    "d": "tokens $D$", "tokens": "tokens $D$",
    "lr": "peak learning rate", "loss": "test loss (nats)",
    "excess": "excess loss $L-L_\\infty$ (nats)", "params": "parameters $N$",
    "width": "embedding dimension $N/512$",
    "tokens_per_param": "tokens per parameter", "flops": "flops", "round": "round",
    "seed": "instance seed", "init": "init scale",
    # the columns a list of IsoFLOP optima carries, so `plot_runs(res.isoflop(), ...)`
    # labels itself like the laws it is about to be fitted into
    "n_star": "optimal size $N^*$", "loss_star": "best loss $L^*$ (nats)",
}


def _decade_ticks(ax):
    """Label a log axis that spans less than a decade.

    ``_finish`` drops minor labels, which is right for a multi-decade axis and leaves a
    narrow one blank -- and an excess-loss axis often covers a factor of 3.  Under two
    decades, put major ticks (which keep their labels) inside the decade instead.
    """
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        if axis.get_scale() != "log":
            continue
        lo, hi = lim
        if lo <= 0:
            continue
        span = hi / lo
        if span < 10:      # sub-decade: plain numbers, since 1.6 reads better than 1.6e0
            axis.set_major_locator(mpl.ticker.LogLocator(subs=(1, 1.5, 2, 3, 5, 7)))
            axis.set_major_formatter(mpl.ticker.ScalarFormatter())
        elif span < 100:   # one or two decades: sci notation, so there is no shared offset
            axis.set_major_locator(mpl.ticker.LogLocator(subs=(1, 2, 5)))
            axis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())


def _rows_of(runs):
    """Accept a Results, a Lab, or a plain list of rows; return its rows.

    No l_inf comes back with them: the problem's irreducible loss is not something you are
    told, so ``excess`` needs a floor you fitted (or know, as in the demo variant) and pass
    in yourself.
    """
    if isinstance(runs, list):
        return runs
    rows = getattr(runs, "rows", None)
    if rows is None:                       # a Lab: take every round it has run
        return list(runs.results.rows)
    return list(rows)


def _column(rows, name, l_inf):
    """One column by name, including the derived ones."""
    if name == "d":
        name = "tokens"
    if name == "width":            # the embedding dimension behind a parameter count
        return np.array([r.get("width", r["n"] / D_OUT) for r in rows], float)
    if name == "params":
        return np.array([r["n"] for r in rows], float)          # n IS the parameter count
    if name == "tokens_per_param":
        return np.array([r["tokens"] / r["n"] for r in rows], float)
    if name == "excess":
        if l_inf is None:
            raise ValueError(
                "excess loss needs a floor: pass l_inf=... (the L_inf your own L*(C) fit "
                "found, or 0.0 on the one-correct-answer demo). The problem does not tell "
                "you its irreducible loss.")
        return np.array([r["loss"] - l_inf for r in rows], float)
    if name not in rows[0]:
        raise KeyError(f"no column {name!r}; have {sorted(set(rows[0]) | set(COLUMNS))}")
    return np.array([r[name] for r in rows], float)


COLORBAR_MAX = 10   # more colour groups than this and a legend stops being readable


def _ramp_cmap():
    """The ordinal ramp as a continuous colormap, for a colour axis with many values."""
    return mcolors.LinearSegmentedColormap.from_list("assocmem", RAMP)


def _color_norm(vals):
    """Log norm across the colour values when they span decades, linear when they do not."""
    v = np.asarray(sorted(vals), float)
    lo, hi = float(v[0]), float(v[-1])
    if lo > 0 and hi / lo >= 10:
        return mcolors.LogNorm(vmin=lo, vmax=hi)
    return mcolors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-12)


# The compute relation is what lets a parametric law be drawn on axes that are not (N, D):
# pin any two of (C, N, D) and the third follows from C = k N D.
_ND_ALIAS = {"tokens": "d", "params": "n"}


def _nd_from(xname, xs, cname, cval, k):
    """(n, d) along a group's x axis, from whichever two of (C, N, D) the axes pin down."""
    xname = _ND_ALIAS.get(xname, xname)
    cname = _ND_ALIAS.get(cname, cname)
    have = {xname: np.asarray(xs, float), cname: np.full(len(xs), float(cval))}
    if set(have) == {"n", "d"}:
        return have["n"], have["d"]
    if set(have) == {"c", "n"}:
        return have["n"], have["c"] / (k * have["n"])
    if set(have) == {"c", "d"}:
        return have["c"] / (k * have["d"]), have["d"]
    raise ValueError(
        f"a parametric law cannot be drawn against x={xname!r} coloured by {cname!r}: "
        f"the pair has to pin down (N, D), so use two of 'n', 'd'/'tokens' and 'c'.")


def plot_runs(runs, x="n", y="loss", color="c", *, excess=False, l_inf=None,
              reduce="min", fit=None, logx=True, logy=True, ax=None, path=None,
              show=None, title=None):
    """Any column of a set of runs against any other, with a third as the colour axis.

    The round figures answer fixed questions (where is lr*, where is the IsoFLOP
    minimum). This answers whichever one you have: ``plot_runs(lab, x="tokens",
    y="loss", color="n")``, ``plot_runs(r1, x="lr", y="excess", color="c")``, and so on
    over :data:`COLUMNS` -- the recorded ones plus ``params``, ``tokens_per_param`` and
    ``excess``.

    excess:  plot ``y - l_inf`` instead of ``y``, with the floor you pass in ``l_inf``.
             Only the excess loss is a power law, so this is what makes a loss axis a
             straight line in log-log -- but the floor is yours to estimate: use what your
             L*(C) fit found, or 0.0 on the one-correct-answer demo, where it is 0 by
             construction.
    color:   the column on the colour axis, or None to draw every run as one series.  Up to
             :data:`COLORBAR_MAX` values get a legend; past that they get a colorbar, since
             fifteen legend entries are not a legend.
    reduce:  'min' keeps the best y per (x, colour) cell -- for a loss axis that is the
             envelope over everything not plotted, e.g. over lr, which is what an
             IsoFLOP curve wants. 'mean' averages; None draws every run.
    fit:     what to overlay on each colour group.
             'powerlaw'  -- a least-squares line through the log-log points, labelled with
                            its slope and r^2.
             'parabola'  -- the IsoFLOP parabola in log x, with its minimum marked: the
                            fit that turns a profile into the (x*, y*) a law is built from.
             a `JointFit` -- the loss curve that parametric law predicts along this group's
                            slice, drawn through the measured points.  Pass the fit you
                            made, so the plot shows the form you chose.
    Both axes are log by default: a power law is only a straight line there.
    """
    rows = _rows_of(runs)
    if not rows:
        raise ValueError("no runs to plot")
    l_inf = None if l_inf is None else float(l_inf)
    if excess:
        if y == "loss":
            y = "excess"
        elif y != "excess":
            raise ValueError(f"excess=True does not apply to y={y!r}")
    parametric = fit if isinstance(fit, _fit.JointFit) else None
    if fit is not None and parametric is None and fit not in ("powerlaw", "parabola"):
        raise ValueError(f"unknown fit {fit!r}; 'powerlaw', 'parabola', a JointFit, or None")
    if parametric is not None and not color:
        raise ValueError("a parametric law needs a colour axis to know which slice of the "
                         "(N, D) plane to draw: colour by 'c', 'n' or 'd'.")

    xv = _column(rows, x, l_inf)
    yv = _column(rows, y, l_inf)
    cv = _column(rows, color, l_inf) if color else None

    groups = []
    for cval in (sorted(set(cv)) if color else [None]):
        m = np.ones(len(rows), bool) if cval is None else (cv == cval)
        xs, ys = xv[m], yv[m]
        if reduce in ("min", "mean"):
            uniq = sorted(set(xs))
            agg = min if reduce == "min" else (lambda v: float(np.mean(v)))
            ys = np.array([agg([ys[i] for i in range(len(xs)) if xs[i] == u])
                           for u in uniq], float)
            xs = np.array(uniq, float)
        else:
            o = np.argsort(xs)
            xs, ys = xs[o], ys[o]
        groups.append((cval, xs, ys))

    # Past COLORBAR_MAX groups the colour axis becomes a bar rather than a list, and the
    # colours come from the value itself instead of from the group's rank.
    bar = color is not None and len(groups) > COLORBAR_MAX
    if bar:
        cmap, norm = _ramp_cmap(), _color_norm([g[0] for g in groups])
        cols = [cmap(norm(g[0])) for g in groups]
    else:
        cols = [_ramp(i, len(groups)) for i in range(len(groups))]

    with mpl.rc_context(STYLE):
        own = ax is None
        fig = plt.figure(figsize=(4.6, 3.6)) if own else ax.figure
        ax = fig.add_subplot(111) if own else ax
        labelled = set()   # in colorbar mode each overlay is named once, not per group

        def once(kind, text):
            """Label the first group's overlay only, when the legend is not per group."""
            if not bar:
                return text
            if kind in labelled:
                return None
            labelled.add(kind)
            return text

        for i, (cval, xs, ys) in enumerate(groups):
            col = cols[i]
            lbl = None if cval is None else (sci(cval) if color == "c" else f"{cval:g}")
            # only join the points when x identifies them: with reduce=None a cell can
            # hold several runs, and a line through them would draw a series that is not
            # one (all the model sizes at one lr, say).
            joined = len(xs) > 1 and len(set(xs)) == len(xs)
            ax.plot(xs, ys, "o-" if joined and parametric is None else "o", ms=5,
                    color=col, label=None if bar else lbl)
            if len(xs) >= 2 and fit == "powerlaw":
                a, b, r2 = _fit.powerlaw(xs, ys)
                xf = np.exp(np.linspace(np.log(xs.min()), np.log(xs.max()), 50))
                ax.plot(xf, a * xf**b, ls="--", lw=1.2, color=col, alpha=.9,
                        label=once("powerlaw",
                                   f"  $\\propto x^{{{b:.2f}}}$, $r^2$={r2:.3f}"))
            elif len(xs) >= 3 and fit == "parabola":
                f = _fit.isoflop_optimum(xs, ys)
                xf = np.exp(np.linspace(np.log(xs.min()), np.log(xs.max()), 100))
                ax.plot(xf, np.polyval(f.coef, np.log(xf)), ls=":", lw=1.2, color=col,
                        alpha=.9, label=once("parabola", "parabola in $\\log x$"))
                ax.plot([f.n_star], [f.loss_star], "v", color=col, ms=8, mec=SURFACE,
                        mew=1.2, zorder=5,
                        label=once("vertex", "$\\blacktriangledown$ fitted optimum"))
            elif len(xs) >= 2 and parametric is not None:
                xf = np.exp(np.linspace(np.log(xs.min()), np.log(xs.max()), 120))
                nn, dd = _nd_from(x, xf, color, cval, parametric.flops_per_nd)
                pred = parametric.predict(nn, dd)
                if excess or y == "excess":
                    pred = pred - (l_inf or 0.0)
                ax.plot(xf, pred, ls="-", lw=1.3, color=col, alpha=.9,
                        label=once("parametric", parametric.label))
        ax.set(xscale="log" if logx else "linear", yscale="log" if logy else "linear",
               xlabel=_AXIS_LABEL.get(x, x), ylabel=_AXIS_LABEL.get(y, y),
               title=title if title is not None else f"{y} vs {x}")
        if bar:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=ax, label=_AXIS_LABEL.get(color, color))
        if ax.get_legend_handles_labels()[1]:
            _headroom(ax, 0.2)
            ax.legend(title=None if bar or not color else _AXIS_LABEL.get(color, color),
                      fontsize=7, title_fontsize=7, labelcolor=INK2,
                      ncol=2 if len(groups) > 4 and not bar else 1)
        _decade_ticks(ax)
        return _finish(fig, path, show) if own else ax


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
                ax.set(xscale="log", xlabel="parameters $N$", ylabel="test loss (nats)",
                       title="runs this round")
            elif name == "nstar":
                cs = [d["c"] for d in iso]
                a, b, r2 = _fit.powerlaw(cs, [d["n_star"] for d in iso])
                panel_law(ax, cs, [d["n_star"] for d in iso], a, b,
                          ylabel="optimal size $N^*$", title="optimal size so far", r2=r2)
            elif name == "loss":
                cs = [d["c"] for d in iso]
                ax.plot(cs, [d["loss_star"] for d in iso], "o-", color=RAMP[3], ms=7,
                        mec=SURFACE, mew=1.2)
                ax.set(xscale="log", xlabel="compute $C$ (flops)",
                       ylabel="best loss $L^*$ (nats)", title="best loss so far")
        fig.suptitle(f"round: {res.name}", fontsize=10, y=1.04)
        return _finish(fig, path, show)


def plot_summary(lab, path=None, show=None):
    """Budget strip + everything measured so far.  The 'where am I' plot.

    Only measurements: the IsoFLOP optima it draws come from the runs, and the one power
    law on it is n*(C) through those optima.  Fitting L*(C) -- picking a form, deciding
    whether the floor is identifiable over your span of rungs -- is the part you do
    yourself, so it is not quietly done for you here.
    """
    res = lab.results
    iso = res.isoflop()
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 6.2))
        gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 3, 3], hspace=0.55, wspace=0.28)
        panel_budget(lab, fig.add_subplot(gs[0, :]))
        panel_isoflop(res, fig.add_subplot(gs[1, 0]), hero=lab.hero_record)
        panel_lr(res, fig.add_subplot(gs[1, 1]))
        ax_curve = fig.add_subplot(gs[1, 2])
        if lab.hero_record:
            r = lab.hero_record
            xs = [s * BATCH for s in r["curve_steps"]]
            ax_curve.plot(xs, r["curve_loss"], "o-", color=ACCENT, ms=5)
            pred = r.get("predicted")
            if pred is not None and np.isfinite(pred):
                ax_curve.axhline(pred, color=INK3, lw=1.2, ls=":")
                ax_curve.text(xs[-1] * 0.16, pred + 0.03, f"predicted {pred:.4f}",
                              color=INK2, fontsize=7.5)
            ax_curve.set(xlabel="tokens $D$", ylabel="test loss (nats)",
                         title=f"c  hero run: {r['loss']:.4f} nats"
                               + (f" (predicted {pred:.4f})"
                                  if pred is not None and np.isfinite(pred) else ""))
        else:
            ax_curve.axis("off")
            ax_curve.text(.5, .5, "hero run not done yet", ha="center", color=INK2)
        cs = [d["c"] for d in iso]
        hc = lab.hero_record["c_train"] if lab.hero_record else None
        ax_n = fig.add_subplot(gs[2, 0])
        if len(iso) >= 2:
            a, b, r2 = _fit.powerlaw(cs, [d["n_star"] for d in iso])
            panel_law(ax_n, cs, [d["n_star"] for d in iso], a, b,
                      ylabel="optimal size $N^*$", title="d  optimal size (measured)",
                      r2=r2, hero=(hc, lab.hero_record["n"]) if lab.hero_record else None)
        else:
            ax_n.axis("off")
            ax_n.text(.5, .5, "one rung so far\n(>=2 rungs for $n^*(C)$)", ha="center",
                      va="center", color=INK2, fontsize=8)
        ax_l = fig.add_subplot(gs[2, 1])
        if iso:
            ax_l.plot(cs, [d["loss_star"] for d in iso], "o-", color=RAMP[3], ms=7,
                      mec=SURFACE, mew=1.2, label="rung optima")
            if lab.hero_record:
                ax_l.plot([hc], [lab.hero_record["loss"]], "*", color=ACCENT, ms=15,
                          mec=SURFACE, mew=1.2, label="hero")
            ax_l.set(xscale="log", xlabel="compute $C$ (flops)",
                     ylabel="best loss $L^*$ (nats)", title="e  best loss (measured)")
            ax_l.legend(fontsize=7, labelcolor=INK2)
        else:
            ax_l.axis("off")
            ax_l.text(.5, .5, "no rung with >=3 sizes yet", ha="center", va="center",
                      color=INK2, fontsize=8)
        ax_t = fig.add_subplot(gs[2, 2])
        ax_t.plot([r["c"] for r in res.rows], [r["tokens"] / r["n"] for r in res.rows],
                  "o", color=RAMP[2], ms=4.5, alpha=.7, label="every run")
        if iso:
            ax_t.plot(cs, [d["c"] / (6.0 * d["n_star"] ** 2) for d in iso], "o-",
                      color=ACCENT, ms=6, mec=SURFACE, mew=1.2, label="at $n^*$")
        ax_t.set(xscale="log", yscale="log", xlabel="compute $C$ (flops)",
                 ylabel="tokens per parameter", title="f  where the budget went")
        ax_t.legend(fontsize=7, labelcolor=INK2)
        fig.suptitle(f"lab '{lab.name}'", fontsize=11, y=0.99)
        return _finish(fig, path, show, tight=False)
