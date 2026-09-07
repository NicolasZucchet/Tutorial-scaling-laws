"""Warm the caches before the tutorial.  `uv run assocmem-prepare`.

There is very little left to do here.  The training stream is generated from a hash of
the stream position, so it needs no file and no preparation; the only things worth
touching in advance are the universal profile table (~0.5 s) and the stratified eval
sets (a few seconds, a few tens of MB), which do get written to disk.

Running this is optional -- everything below happens on first use anyway.
"""

from __future__ import annotations

import time

from .data import _profiles
from .grid import strat_evalset
from .lab import CHECK_STRAT, HERO_STRAT, SCREEN_STRAT
from .problem import TOTAL_TOKENS, get_stream


def main() -> None:
    jobs = [("conditional profile table", lambda: _profiles()),
            ("eval set (screening)", lambda: strat_evalset(**SCREEN_STRAT)),
            ("eval set (hero)", lambda: strat_evalset(**HERO_STRAT)),
            ("eval set (resampled tail)", lambda: strat_evalset(**CHECK_STRAT)),
            (f"training stream ({TOTAL_TOKENS:,} tokens, not cached)",
             lambda: get_stream(TOTAL_TOKENS))]
    for name, f in jobs:
        t0 = time.time()
        f()
        print(f"  {name:<48s} {time.time() - t0:6.1f} s")
    # The irreducible loss is deliberately NOT printed: the exercise is to find it, and a
    # student who runs this would otherwise be handed the answer.
    print("\nready.  predicting uniform would give log(512) = 6.2383 nats")


if __name__ == "__main__":
    main()
