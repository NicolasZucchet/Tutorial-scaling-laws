"""Build the cached data once, before the tutorial.  `uv run assocmem-prepare`.

Generating the Zipf stream and the eval-set conditionals takes ~1 minute and ~250 MB;
doing it up front means the first `run_round` is pure training.
"""

from __future__ import annotations

import time

from .lab import EVAL_TOKENS, HERO_CHECK_TOKENS, HERO_EVAL_TOKENS
from .problem import MASTER_TOKENS, get_evalset, get_stream


def main() -> None:
    jobs = [("training stream", lambda: get_stream(MASTER_TOKENS)),
            (f"eval set ({EVAL_TOKENS} tokens, screening)", lambda: get_evalset(EVAL_TOKENS)),
            (f"eval set ({HERO_EVAL_TOKENS} tokens, hero)", lambda: get_evalset(HERO_EVAL_TOKENS)),
            (f"eval set ({HERO_CHECK_TOKENS} tokens, held-out)",
             lambda: get_evalset(HERO_CHECK_TOKENS, seed=1))]
    for name, f in jobs:
        t0 = time.time()
        f()
        print(f"  {name:<44s} {time.time() - t0:6.1f} s")
    ev = get_evalset(EVAL_TOKENS)
    print(f"\nready.  irreducible loss (mean conditional entropy) = "
          f"{ev.entropy.mean():.4f} nats;  uniform baseline = 6.2383 nats")


if __name__ == "__main__":
    main()
