"""Figures for the (N, D) scan: measured scaling law vs the envelope calculation.

    PYTHONPATH=src uv run python scripts/grid_figures.py

Writes figures/grid_law.png (six panels: the two axes, the local exponents, the
extrapolation test, and the per-context loss that explains all of it) and
figures/grid_map.png (the grid itself, and the compute-optimal frontier).
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from assocmem.data import D_OUT, GAMMA
from assocmem.grid import build_strat_eval
from assocmem.grid_fit import (A_TH, B_TH, CAP_PER_H, cells_of, envelope, fit_law,
                               grid_means, isoflop_profiles, l_measured, local_slopes)
from assocmem.plots import ACCENT, GRID, INK, INK2, INK3, RAMP, STYLE, SURFACE, _ramp, sci

ROOT = pathlib.Path(__file__).resolve().parents[1]
A_HS = (32, 64, 128, 256, 512)
A_STEPS = (100, 400, 1600, 6400, 25_600)
# Sequential = one hue, light -> dark, taken from the deck's own ramp.
SEQ = LinearSegmentedColormap.from_list("assoc", RAMP)


def _annotated(ax, mat, xlab, ylab, xt, yt, title, target, vmax=None):
    """A small annotated heatmap: the numbers are the point, colour is the guide."""
    m = np.ma.masked_invalid(mat)
    im = ax.imshow(m, origin="lower", cmap=SEQ, vmin=0,
                   vmax=vmax or np.nanmax(mat) * 1.02, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            near = abs(mat[i, j] / target - 1) < 0.06
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6.6,
                    color="w" if mat[i, j] > 0.62 * (vmax or np.nanmax(mat)) else INK,
                    fontweight="bold" if near else "normal")
    ax.set(xticks=range(len(xt)), yticks=range(len(yt)), xlabel=xlab, ylabel=ylab,
           title=title)
    ax.set_xticklabels(xt, fontsize=6.6)
    ax.set_yticklabels(yt, fontsize=6.6)
    ax.grid(False)
    return im


def main() -> None:
    store = json.loads((ROOT / "results/grid.json").read_text())
    se = build_strat_eval()
    l = store["meta"]["l"]
    l_hat = l_measured(store)
    m, dn, dd = local_slopes(store)
    n, d, y, *_ = cells_of(store, A_HS, A_STEPS)
    law = fit_law(n, d, y, form="power_mean")
    add = fit_law(n, d, y, form="additive")
    held = sorted(k for k in m if k[0] not in A_HS or k[1] not in A_STEPS)
    hs_all, ss_all = sorted({k[0] for k in m}), sorted({k[1] for k in m})

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(2, 3, figsize=(13.6, 7.7))
        (a, b, c), (e, f, g) = ax

        # (a) the model axis ----------------------------------------------
        for i, s in enumerate(ss_all):
            pts = sorted((h, m[(h, s)]) for h in hs_all if (h, s) in m)
            if len(pts) < 2:
                continue
            a.plot([D_OUT * h for h, _ in pts], [v for _, v in pts], "o-", ms=4.5,
                   lw=1.6, color=_ramp(i, len(ss_all)), label=sci(64 * s))
        anc = sorted((h, m[(h, ss_all[-1])]) for h in hs_all if (h, ss_all[-1]) in m)
        x0, y0 = D_OUT * anc[0][0], anc[0][1]
        xr = np.array([x0, D_OUT * hs_all[-1] * 1.6])
        a.plot(xr, y0 * (xr / x0) ** -A_TH, ls="--", lw=1.5, color=INK)
        a.text(xr[-1], y0 * (xr[-1] / x0) ** -A_TH,
               r"  $\propto N^{-(\alpha-1)}$", color=INK, fontsize=8, va="center")
        a.set(xscale="log", yscale="log", xlabel="parameters  $N=512h$",
              ylabel="$L-L_\\infty$  (nats)",
              title="(a) model axis: steepest where $N$ binds")
        a.legend(title="tokens $D$", fontsize=6.4, title_fontsize=6.4, ncol=2,
                 labelcolor=INK2, loc="lower left")

        # (b) the data axis -----------------------------------------------
        for i, h in enumerate(hs_all):
            pts = sorted((s, m[(h, s)]) for s in ss_all if (h, s) in m)
            if len(pts) < 2:
                continue
            b.plot([64 * s for s, _ in pts], [v for _, v in pts], "o-", ms=4.5, lw=1.6,
                   color=_ramp(i, len(hs_all)), label=f"$h$={h}")
        hb = max(h for h in hs_all if sum((h, s) in m for s in ss_all) >= 2)
        anc = sorted((s, m[(hb, s)]) for s in ss_all if (hb, s) in m)
        x0, y0 = 64 * anc[0][0], anc[0][1]
        xr = np.array([x0, 64 * ss_all[-1] * 2.2])
        b.plot(xr, y0 * (xr / x0) ** -B_TH, ls="--", lw=1.5, color=INK)
        b.text(xr[-1], y0 * (xr[-1] / x0) ** -B_TH,
               r"  $\propto D^{-(1-1/\alpha)}$", color=INK, fontsize=8, va="center")
        b.set(xscale="log", yscale="log", xlabel="training tokens  $D$ (single pass)",
              ylabel="$L-L_\\infty$  (nats)",
              title="(b) data axis: steepest where $D$ binds")
        b.legend(fontsize=6.4, ncol=2, labelcolor=INK2, loc="lower left")

        # (c)+(e) local exponents ----------------------------------------
        pairs_h = [(x, y_) for x, y_ in zip(hs_all[:-1], hs_all[1:])]
        mat = np.full((len(ss_all), len(pairs_h)), np.nan)
        for i, s in enumerate(ss_all):
            for j, (h1, h2) in enumerate(pairs_h):
                if (h1, h2, s) in dn:
                    mat[i, j] = dn[(h1, h2, s)]
        _annotated(c, mat, "$h$ pair", "tokens $D$",
                   [f"{x}:{y_}" for x, y_ in pairs_h], [sci(64 * s) for s in ss_all],
                   r"(c) local $-d\log E/d\log N$" "\n"
                   f"bold: within 6 % of $\\alpha-1={A_TH:.2f}$", A_TH, vmax=0.26)
        pairs_s = [(x, y_) for x, y_ in zip(ss_all[:-1], ss_all[1:])]
        mat2 = np.full((len(hs_all), len(pairs_s)), np.nan)
        for i, h in enumerate(hs_all):
            for j, (s1, s2) in enumerate(pairs_s):
                if (h, s1, s2) in dd:
                    mat2[i, j] = dd[(h, s1, s2)]
        _annotated(e, mat2, "$D$ pair", "width $h$",
                   [f"{sci(64 * x)}" for x, _ in pairs_s], [str(h) for h in hs_all],
                   r"(d) local $-d\log E/d\log D$" "\n"
                   f"bold: within 6 % of $1-1/\\alpha={B_TH:.3f}$", B_TH, vmax=0.26)

        # (f) extrapolation ----------------------------------------------
        for lab, lw_, col, mk in (("additive (Chinchilla) form", add, INK3, "s"),
                                  ("power mean, $q$ fitted", law, RAMP[4], "o")):
            f.plot(y, lw_.predict(n, d), mk, ms=5, color=col, alpha=.8, mec="none",
                   label=f"{lab}  (fit region)")
        if held:
            nh = np.array([D_OUT * k[0] for k in held], float)
            dh = np.array([64 * k[1] for k in held], float)
            yh = np.array([m[k] for k in held])
            f.plot(yh, add.predict(nh, dh), "s", ms=9, color=INK, mec=SURFACE, mew=1.4,
                   label="additive, extrapolated")
            f.plot(yh, law.predict(nh, dh), "o", ms=9, color=ACCENT, mec=SURFACE,
                   mew=1.4, label="power mean, extrapolated")
        lim = [0.3, 2.1]
        f.plot(lim, lim, lw=1.0, color=INK3, zorder=0)
        for k in (0.9, 1.1):
            f.plot(lim, [k * v for v in lim], lw=0.9, ls=":", color=GRID, zorder=0)
        f.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel="measured  $L-L_\\infty$", ylabel="predicted",
              xticks=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0], yticks=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
              title="(e) extrapolation out of the fit region\n$\\pm$10 % dotted")
        for axis in (f.xaxis, f.yaxis):
            axis.set_major_formatter(mpl.ticker.ScalarFormatter())
            axis.set_minor_formatter(mpl.ticker.NullFormatter())
        f.legend(fontsize=6.2, labelcolor=INK2, loc="upper left")

        # (g) the per-context loss the envelope idealises as a step -------
        show = [(64, 400), (64, 25_600), (512, 1600), (512, 409_600), (2048, 409_600)]
        show = [k for k in show if f"h{k[0]}/s{k[1]}/z0" in store["cells"]]
        for i, (h, s) in enumerate(show):
            pb = np.array(store["cells"][f"h{h}/s{s}/z0"]["per_bin"])
            col = _ramp(i, len(show))
            g.plot(se.bin_lo, pb, lw=1.7, color=col, label=f"$h$={h}, $D$={sci(64 * s)}")
            k = min(CAP_PER_H * h, (64 * s) ** (1 / GAMMA))
            g.plot([k], [0.12], "^", ms=9, color=col, mec=SURFACE, mew=1.0,
                   clip_on=False, zorder=5)
        g.axhline(l_hat, color=INK3, lw=1.2, ls="--")
        g.text(1.15, l_hat, f" measured, {l_hat:.2f}", color=INK2, fontsize=6.8,
               va="bottom", ha="left")
        g.axhline(l, color=INK3, lw=1.0, ls=":")
        g.text(1.15, l, f" assumed $l={l:.2f}$", color=INK2, fontsize=6.8,
               va="top", ha="left")
        g.set(xscale="log", ylim=(0, 5.3),
              xlabel="context index $i$   ($p(i)\\propto i^{-\\alpha}$)",
              ylabel="excess loss of context $i$  (nats)",
              title="(f) the assumed step, measured\n"
                    "$\\blacktriangle$ = $\\min($capacity$(N),\\,D^{1/\\alpha})$")
        g.legend(fontsize=6.2, labelcolor=INK2, loc="lower right")

        fig.suptitle("Associative memory at $\\alpha=1.2$: the measured scaling law vs "
                     "the back-of-the-envelope prediction", fontsize=11, y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(ROOT / "figures/grid_law.png", dpi=190)
        print("wrote figures/grid_law.png")

        # ------------------------------------------------------------ fig 2
        fig2, ax2 = plt.subplots(1, 2, figsize=(11.4, 4.4))
        p, q = ax2
        # true log axes, so the iso-FLOP lines and the ridge are straight
        def edges(v):
            v = np.asarray(v, float)
            mid = np.sqrt(v[:-1] * v[1:])
            return np.concatenate([[v[0] ** 2 / mid[0]], mid, [v[-1] ** 2 / mid[-1]]])

        nx, dy = D_OUT * np.array(hs_all, float), 64 * np.array(ss_all, float)
        z = np.full((len(ss_all), len(hs_all)), np.nan)
        for (h, s_) in m:
            z[ss_all.index(s_), hs_all.index(h)] = m[(h, s_)]
        im = p.pcolormesh(edges(nx), edges(dy), np.ma.masked_invalid(z), cmap=SEQ,
                          shading="flat")
        plt.colorbar(im, ax=p, label="$L-L_\\infty$  (nats)")
        for (h, s_), v in m.items():
            p.text(D_OUT * h, 64 * s_, f"{v:.2f}", ha="center", va="center",
                   fontsize=6.2, color="w" if v > 1.42 else INK)
        for h, s_ in held:
            p.plot(D_OUT * h, 64 * s_, "s", ms=19, mfc="none", mec=ACCENT, mew=1.5)
        xs = np.array([edges(nx)[0], edges(nx)[-1]])
        for cc in (1e10, 1e11, 1e12, 1e13, 1e14):
            p.plot(xs, cc / (6 * xs), lw=0.9, ls=":", color=INK3, zorder=3)
        ridge = np.array([law.optimum(cc)[:2] for cc in np.logspace(9.0, 15.0, 240)])
        keep = ((ridge[:, 0] >= edges(nx)[0]) & (ridge[:, 0] <= edges(nx)[-1])
                & (ridge[:, 1] >= edges(dy)[0]) & (ridge[:, 1] <= edges(dy)[-1]))
        ridge = ridge[keep]
        p.plot(ridge[:, 0], ridge[:, 1], color=ACCENT, lw=2.2, zorder=4)
        p.text(ridge[-1, 0], ridge[-1, 1], "compute-optimal ridge  ", color=ACCENT,
               fontsize=7.5, ha="right", va="bottom", fontweight="bold", zorder=5)
        p.set(xscale="log", yscale="log", xlim=(edges(nx)[0], edges(nx)[-1]),
              ylim=(edges(dy)[0], edges(dy)[-1]),
              xlabel="parameters  $N=512h$", ylabel="tokens  $D$",
              title="excess loss over the grid;  dotted: iso-FLOP\n"
                    "orange squares: held out from the fit")
        p.grid(False)
        for cc in (1e10, 1e11, 1e12, 1e13, 1e14):  # label where each line exits the box
            xx = cc / (6 * edges(dy)[-1])
            if edges(nx)[0] < xx < edges(nx)[-1]:
                p.text(xx, edges(dy)[-1], f" {sci(cc)}", fontsize=6, color=INK3,
                       va="top", ha="left")

        cs = np.logspace(9.5, 14.6, 90)
        q.plot(cs, [law.optimum(cc)[2] for cc in cs], color=RAMP[4], lw=1.9,
               label=f"power-mean fit:  $C^{{-{law.loss_exponent:.4f}}}$")
        q.plot(cs, [add.optimum(cc)[2] for cc in cs], color=INK3, lw=1.4, ls="-.",
               label=f"additive fit:  $C^{{-{add.loss_exponent:.4f}}}$")
        pref = law.optimum(1e12)[2] / 1e12 ** -(A_TH * B_TH / (A_TH + B_TH))
        q.plot(cs, pref * cs ** -(A_TH * B_TH / (A_TH + B_TH)), ls="--", lw=1.5,
               color=INK, label=f"theory:  $C^{{-{A_TH * B_TH / (A_TH + B_TH):.4f}}}$")
        profs = [x for x in isoflop_profiles(store) if not x["edge"]]
        if profs:
            q.plot([x["c"] for x in profs], [x["excess_star"] for x in profs], "o",
                   ms=8, color=ACCENT, mec=SURFACE, mew=1.3,
                   label="IsoFLOP minima (measured)")
        for cc, loss in ((5.009e12, 3.2993), (6.971e12, 3.2765), (8.826e12, 3.2609)):
            q.plot([cc], [loss - 2.4632], "*", ms=13, color=INK, mec=SURFACE, mew=1.0)
        q.text(1.05e13, 3.2609 - 2.4632, " the three $10^{13}$\n hero runs", color=INK2,
               fontsize=6.8, va="center")
        q.set(xscale="log", yscale="log", xlabel="compute  $C=6ND$  (flops)",
              ylabel="$L^*-L_\\infty$  (nats)", title="compute-optimal frontier")
        q.legend(fontsize=6.6, labelcolor=INK2, loc="lower left")

        fig2.tight_layout()
        fig2.savefig(ROOT / "figures/grid_map.png", dpi=190)
        print("wrote figures/grid_map.png")


if __name__ == "__main__":
    main()
