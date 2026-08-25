"""Figure for the alpha sweep: do the exponents track the tail exponent?

    PYTHONPATH=src uv run python scripts/alpha_figures.py   ->  figures/alpha_sweep.png
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from assocmem.plots import ACCENT, INK, INK2, INK3, RAMP, STYLE, SURFACE

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAVY, RED = "#0f3460", "#c0392b"  # the deck's colours: navy = model size, red = data


def main() -> None:
    rows = json.loads((ROOT / "results/alpha_sweep.json").read_text())
    al = np.array([r["alpha"] for r in rows])
    a = np.array([r["a"] for r in rows])
    b = np.array([r["b"] for r in rows])
    g = np.linspace(min(al.min(), 1.05) - 0.02, al.max() + 0.05, 200)

    with mpl.rc_context(STYLE):
        fig, (p, q) = plt.subplots(1, 2, figsize=(10.6, 4.1))

        p.plot(g, g - 1.0, ls="--", lw=1.6, color=NAVY, label=r"predicted $\alpha-1$")
        p.plot(g, 1.0 - 1.0 / g, ls="--", lw=1.6, color=RED,
               label=r"predicted $1-1/\alpha$")
        # The model axis can only be read where the model is what binds: D^(1/alpha)
        # must run well past the capacity, and the margin needed grows with alpha.
        masked = [r for r in rows if "a_ext" in r]
        if masked:
            p.plot([r["alpha"] for r in masked], [r["a"] for r in masked], "o", ms=8,
                   mfc="none", color=NAVY, mew=1.5,
                   label="model axis, data-masked corner")
            for r in masked:
                if abs(r["a_ext"] - r["a"]) < 0.02:
                    continue  # nothing was masked here; an arrow would just be clutter
                p.annotate("", xy=(r["alpha"], r["a_ext"]), xytext=(r["alpha"], r["a"]),
                           arrowprops=dict(arrowstyle="->", color=INK3, lw=1.0,
                                           shrinkA=5, shrinkB=5))
                p.text(r["alpha"] + 0.014, 0.5 * (r["a"] + r["a_ext"]),
                       f"16$\\times$ more $D$", fontsize=6.2, color=INK3, va="center")
        best = [r.get("a_ext", r["a"]) for r in rows]
        p.plot(al, best, "o", ms=8, color=NAVY, mec=SURFACE, mew=1.4,
               label="model axis, where $N$ binds")
        p.plot(al, b, "s", ms=7.5, color=RED, mec=SURFACE, mew=1.4,
               label="data axis, where $D$ binds")
        p.set(xlabel=r"tail exponent $\alpha$", ylabel="loss exponent",
              title="(a) both exponents are functions of the tail\n"
                    "hollow: corner still masked by the other constraint")
        p.legend(fontsize=6.6, labelcolor=INK2, loc="upper left")

        have = [r for r in rows if "pn" in r]
        if have:
            ah = np.array([r["alpha"] for r in have])
            pn = np.array([r["pn"] for r in have])
            pl = np.array([-r["pl"] for r in have])
            at, bt = g - 1.0, 1.0 - 1.0 / g
            q.plot(g, bt / (at + bt), ls="--", lw=1.6, color=NAVY,
                   label=r"predicted $N^*\propto C^{b/(a+b)}$")
            q.plot(g, at * bt / (at + bt), ls="--", lw=1.6, color=RED,
                   label=r"predicted $L^*\propto C^{-ab/(a+b)}$")
            q.plot(ah, pn, "o", ms=8, color=NAVY, mec=SURFACE, mew=1.4,
                   label="measured $N^*$ (IsoFLOP)")
            q.plot(ah, pl, "s", ms=7.5, color=RED, mec=SURFACE, mew=1.4,
                   label="measured $L^*$ (IsoFLOP)")
        q.set(xlabel=r"tail exponent $\alpha$", ylabel="compute-optimal exponent",
              title="(b) and so is the compute-optimal split")
        q.legend(fontsize=7, labelcolor=INK2, loc="center left")

        fig.suptitle("The exponents come from the tail of the data, not the "
                     "architecture", fontsize=10.5, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        fig.savefig(ROOT / "figures/alpha_sweep.png", dpi=190)
        print("wrote figures/alpha_sweep.png")
        for r in rows:
            ext = f" -> {r['a_ext']:.4f} at {r['a_ext_margin']:.0f}x margin" \
                if "a_ext" in r else ""
            print(f"  alpha={r['alpha']:.2f}  a={r['a']:.4f}{ext} "
                  f"(pred {r['a_th']:.4f})  b={r['b']:.4f} (pred {r['b_th']:.4f})")


if __name__ == "__main__":
    main()
