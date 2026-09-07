"""Compute ledger.  Every run appends here so the 1e13 flop budget is auditable."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

BUDGET = 1e13
ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path(os.environ.get("ASSOCMEM_LEDGER", ROOT / "results" / "ledger.jsonl"))


def configure(path=None, budget: float | None = None) -> None:
    """Point the ledger at another file / budget (used by `Lab` to scope accounting)."""
    global LEDGER, BUDGET
    if path is not None:
        LEDGER = Path(path)
    if budget is not None:
        BUDGET = float(budget)


def log(tag: str, *, train: float = 0.0, eval: float = 0.0, eval_free: float = 0.0,
        **info) -> None:
    """Append one record.  ``eval_free`` is evaluation that is measured but not charged:
    it is reported separately and never counts against the budget (see `train_sweep`)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(t=time.time(), tag=tag, train=float(train), eval=float(eval),
               eval_free=float(eval_free), **info)
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def records() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def total() -> dict:
    recs = records()
    tr = sum(r["train"] for r in recs)
    ev = sum(r["eval"] for r in recs)
    free = sum(r.get("eval_free", 0.0) for r in recs)
    return dict(train=tr, eval=ev, eval_free=free, total=tr + ev,
                remaining=BUDGET - tr - ev, n_runs=len(recs))


def report() -> str:
    t = total()
    by_tag: dict[str, float] = {}
    free_by_tag: dict[str, float] = {}
    for r in records():
        by_tag[r["tag"]] = by_tag.get(r["tag"], 0.0) + r["train"] + r["eval"]
        free_by_tag[r["tag"]] = free_by_tag.get(r["tag"], 0.0) + r.get("eval_free", 0.0)
    lines = [f"budget {BUDGET:.3g} | spent {t['total']:.4g} "
             f"({100 * t['total'] / BUDGET:.2f}%) | remaining {t['remaining']:.4g}",
             f"  train {t['train']:.4g}   eval {t['eval']:.4g}   ({t['n_runs']} sweeps)"]
    if t["eval_free"]:
        lines.append(f"  plus {t['eval_free']:.4g} of uncharged evaluation "
                     f"(the final run's scoring, not billed to the budget)")
    for k, v in sorted(by_tag.items(), key=lambda kv: -kv[1]):
        free = free_by_tag.get(k, 0.0)
        if not v and not free:
            continue
        lines.append(f"    {k:<28s} {v:.4g}  ({100 * v / BUDGET:.2f}%)"
                     + (f"   + {free:.4g} uncharged" if free else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
