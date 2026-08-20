"""Scaling laws for a linear-softmax associative memory.

Tutorial API (see `assocmem.lab`)
--------------------------------
    from assocmem import Lab, Sweep
    lab = Lab("alice")
    s = Sweep(c=[4e9, 1.2e10], n=[64, 128, 256, 512], lr=[0.03, 0.06, 0.12])
    s.estimate(lab)                      # free
    r = lab.run_round("lr landscape", s) # spends one round, plots itself
    laws = lab.fit()
    lab.hero(laws)

Low-level API
-------------
    from assocmem import get_stream, get_evalset, train_sweep

    stream = get_stream(64 * 2000)
    evals  = get_evalset(4096)
    res = train_sweep(n=512, steps=2000, lrs=[1e-3, 3e-3, 1e-2],
                      stream=stream, eval_set=evals)
    print(res.loss, res.best())
"""

from .data import D_OUT, GAMMA, P_CUT, vocab_size, zipf_norm
from .lab import BudgetError, Lab, Laws, Results, Sweep, fit_laws
from .ledger import BUDGET
from .ledger import log as log_flops
from .ledger import report as flop_report
from .ledger import total as flops_spent
from .problem import get_evalset, get_stream
from .train import (BATCH, EvalSet, Stream, evaluate, eval_flops, plan_cost, train_flops,
                    train_sweep)

__all__ = [
    "Lab", "Sweep", "Results", "Laws", "fit_laws", "BudgetError",
    "D_OUT", "GAMMA", "P_CUT", "BATCH", "BUDGET",
    "vocab_size", "zipf_norm", "get_stream", "get_evalset",
    "train_sweep", "train_flops", "eval_flops", "Stream", "EvalSet",
    "log_flops", "flop_report", "flops_spent", "plan_cost", "evaluate",
]
