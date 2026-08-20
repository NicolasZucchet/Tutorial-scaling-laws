"""The hero run.  One shot, sized to whatever flops are left in the ledger.

Recipe comes entirely from results/final_fit.json:
    n      = a_n * C^b_n            (IsoFLOP optimum)
    steps  = C / (6 * 512 * n * 64)
    lr_max = a_lr * C^p_lr          (cosine to lr_max/10, zero init, Adam, batch 64)
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np

from assocmem import fit, ledger
from assocmem.problem import get_evalset, get_stream
from assocmem.train import BATCH, evaluate, eval_flops, plan_cost, train_flops, train_sweep

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
CURVE_POINTS, CURVE_TOKENS = 6, 2048
FINAL_TOKENS = 65536  # headline eval set (seed 0)
CHECK_TOKENS = 32768  # independent eval set (seed 1)
MARGIN = 2.0e10

ff = json.loads((RESULTS / "final_fit.json").read_text())
a_n, b_n = ff["n_law"]["a"], ff["n_law"]["b"]
a_lr, p_lr = ff["lr_law"]["a"], ff["lr_law"]["p"]

# ---- size the run so that train + all evals fit exactly in what is left -------
left = ledger.total()["remaining"]
c = left - MARGIN
for _ in range(50):  # eval cost depends on n, which depends on c: iterate to a fixed point
    n = int(round(a_n * c**b_n / 8) * 8)
    ev = eval_flops(n, CURVE_TOKENS) * CURVE_POINTS + eval_flops(n, FINAL_TOKENS) \
        + eval_flops(n, CHECK_TOKENS)
    c_new = left - MARGIN - ev
    if abs(c_new - c) < 1e6:
        break
    c = c_new
steps = fit.steps_for(c, n)
c_train = train_flops(n, steps)
lr = a_lr * c_train**p_lr
pred_pinned = ff["l_inf"] + ff["l_law"]["a"] * c_train ** ff["l_law"]["b"]
pred_free = ff["l_law_free"]["l_inf"] + ff["l_law_free"]["a"] * c_train ** -ff["l_law_free"]["alpha"]

print(f"budget left {left:.5g}")
print(f"  n      = {n}          (N = 512*n = {512 * n:,} params)")
print(f"  steps  = {steps}      (D = {steps * BATCH:,} tokens)")
print(f"  lr_max = {lr:.5f}     lr_min = {lr / 10:.6f}, cosine; zero init; Adam; batch {BATCH}")
print(f"  train flops {c_train:.5g} + eval {ev:.4g} = {c_train + ev:.5g}  "
      f"(margin {left - c_train - ev:.3g})")
print(f"  PREDICTED loss = {pred_pinned:.4f} [pinned L_inf] / {pred_free:.4f} [free]")
assert c_train + ev <= left, "over budget"

stream = get_stream(4_000_000)
ev_final = get_evalset(FINAL_TOKENS, seed=0)
ev_check = get_evalset(CHECK_TOKENS, seed=1)
ev_curve = get_evalset(4096, seed=0)

print("\ntraining...", flush=True)
res = train_sweep(n=n, steps=steps, lrs=[lr], stream=stream, eval_set=ev_curve,
                  eval_tokens=CURVE_TOKENS, eval_points=CURVE_POINTS,
                  eval_chunk=CURVE_TOKENS, tag="hero", return_params=True)
for s, l in zip(res.curve_steps, res.curve.ravel()):
    print(f"   step {s:7d}  loss {l:.4f}")

exact0, samp0, m0 = evaluate(res.params, ev_final, n=n, instance_seed=0, y_seed=11)
exact1, samp1, m1 = evaluate(res.params, ev_check, n=n, instance_seed=0, y_seed=12)
ledger.log("hero-final-eval", eval=eval_flops(n, m0) + eval_flops(n, m1), n=n)

print(f"\n=== HERO RESULT (n={n}, steps={steps}, lr={lr:.5f}) ===")
print(f"  eval set A ({m0} tokens, seed 0): exact CE = {exact0[0]:.4f}   "
      f"sampled CE = {samp0[0]:.4f}   irreducible = {ev_final.entropy[:m0].mean():.4f}")
print(f"  eval set B ({m1} tokens, seed 1): exact CE = {exact1[0]:.4f}   "
      f"sampled CE = {samp1[0]:.4f}   irreducible = {ev_check.entropy[:m1].mean():.4f}")
print(f"  PREDICTED {pred_pinned:.4f} / {pred_free:.4f}   ->  ACTUAL {exact0[0]:.4f}  "
      f"(err {exact0[0] - pred_pinned:+.4f} / {exact0[0] - pred_free:+.4f})")

(RESULTS / "hero.json").write_text(json.dumps(dict(
    n=n, steps=steps, tokens=steps * BATCH, lr_max=lr, lr_min=lr / 10, batch=BATCH,
    params=512 * n, c_train=c_train, c_eval=ev,
    predicted_pinned=pred_pinned, predicted_free=pred_free,
    loss_exact_A=float(exact0[0]), loss_sampled_A=float(samp0[0]), n_eval_A=int(m0),
    loss_exact_B=float(exact1[0]), loss_sampled_B=float(samp1[0]), n_eval_B=int(m1),
    irreducible_A=float(ev_final.entropy[:m0].mean()),
    irreducible_B=float(ev_check.entropy[:m1].mean()),
    curve_steps=[int(x) for x in res.curve_steps],
    curve_loss=[float(x) for x in res.curve.ravel()]), indent=1))
np.save(RESULTS / "hero_W.npy", np.asarray(res.params[0]))
print("\n" + ledger.report())
