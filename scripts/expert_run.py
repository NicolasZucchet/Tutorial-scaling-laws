"""A third attempt, designed with everything the first two taught us.

The two earlier attempts differed only in how much they spent on screening:

    careful (scripts/round*.py)   47 % on tuning -> hero 3.2993
    notebook defaults            26 % on tuning -> hero 3.2765

so the binding constraint is not law quality, it is flops. Quantitatively, with
excess loss ~ C^-0.097 and the hero getting whatever is left, **1e12 spent on
screening costs ~0.008 nats**, while the two mis-tuning penalties are

    n  off by a factor g:  0.058 * ln(g)^2     (20 % off  -> 0.0014 nats)
    lr off by a factor f:  0.083 * ln(f)^2     (1.3x off  -> 0.006  nats)

So: get lr right, do not bother getting n right, and screen as cheaply as the
laws allow. Three tricks make the rounds ~7x cheaper than the careful attempt:

1. **Sweep lr at one width per rung.** lr* was independent of n at every rung of
   both earlier studies, so a 3-point lr parabola costs 2 extra runs, not 2 per n.
2. **Build the ladder downwards.** A rung costs (#configs x C), so span is far
   cheaper to buy at the bottom (2e9) than at the top. Extrapolating 90x instead
   of 17x hurts the *prediction*, not the *score*.
3. **Keep every screening run >= ~200 steps.** Below that the cosine schedule is
   in a different regime and lr* stops extrapolating, which would break trick 1.
   That sets the floor rung at ~2e9.

Grids are centred on the previous studies' laws (n* = 1.28e-3 C^0.493,
lr* = 14.9 C^-0.238) -- this run is an informed re-run, not a blind one, and the
report says so.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import json

from assocmem import Lab, Sweep
from assocmem.plots import plot_summary

# priors from the two earlier studies, used only to *centre* the grids
N_STAR = lambda c: 1.282e-3 * c**0.4929
LR_STAR = lambda c: 14.85 * c**-0.2384
SPREAD = 1.75  # lr parabola bracket


def rung(c, ns, lr_at):
    """IsoFLOP profile at compute `c` plus a 3-point lr parabola at width `lr_at`."""
    lr = LR_STAR(c)
    return (Sweep(c=[c], n=ns, lr=[lr])
            + Sweep(c=[c], n=[lr_at], lr=[lr / SPREAD, lr * SPREAD]))


lab = Lab("expert", budget=1e13, rounds=3,
          hero_curve_points=4, hero_eval_tokens=65536, hero_check_tokens=32768)

ROUNDS = [
    ("R1 two cheap rungs",
     rung(2.0e9, [16, 32, 64, 128, 256], 64) + rung(1.0e10, [32, 64, 128, 256, 512], 128)),
    ("R2 third rung", rung(3.0e10, [64, 128, 256, 512], 256)),
    ("R3 top rung", rung(1.0e11, [160, 320, 640], 320)),
]

for name, sweep in ROUNDS:
    print(f"\n{'=' * 78}\n{name}   (n* prior {N_STAR(sweep.configs[0].c):.0f}, "
          f"lr prior {LR_STAR(sweep.configs[0].c):.4f})\n{'=' * 78}")
    sweep.estimate(lab)
    lab.run_round(name, sweep, plot=True)

print(f"\n{'=' * 78}\nLAWS\n{'=' * 78}")
laws = lab.fit()
print(f"\nscreening total: {lab.spent:.4g} flops ({100 * lab.spent / lab.budget:.1f}% "
      f"of budget) -- careful attempt spent 47%, notebook defaults 26%")

print(f"\n{'=' * 78}\nHERO\n{'=' * 78}")
hero = lab.hero(laws)
plot_summary(lab, laws, path=lab.dir / "summary.png", show=False)

prev = {"careful (scripts/)": 3.2993, "notebook defaults": 3.2765}
print(f"\n{'=' * 78}\nSCOREBOARD (same eval set, same problem instance)\n{'=' * 78}")
rows = sorted([*prev.items(), ("this expert re-run", hero["loss"])], key=lambda kv: kv[1])
for k, v in rows:
    print(f"  {k:<22s} {v:.4f} nats" + ("   <-- best" if v == rows[0][1] else ""))
json.dump(dict(laws=dict(n_law=laws.n_law, lr_law=laws.lr_law, loss_law=laws.loss_law,
                         l_inf=laws.l_inf, rungs=laws.rungs, notes=laws.notes),
               hero=hero, screening_flops=lab.spent - hero["c_train"] - hero["c_eval"],
               scoreboard=dict(rows)),
          open(lab.dir / "expert_run.json", "w"), indent=1, default=float)
print(f"\n{lab.status()}")
