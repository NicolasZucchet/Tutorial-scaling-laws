"""Build the appended stream segment, tokens 4M -> 26.2M.  One-time, ~5 min, ~265 MB.

Writes `results/cache/stream_ext_s0.npz` and touches nothing else: the canonical
4M-token base file stays byte-for-byte as it is, so every result already recorded
against it remains valid.  See the comment block at the top of assocmem/problem.py.

    PYTHONPATH=src uv run python scripts/build_extension.py
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assocmem import problem as P  # noqa: E402


def main() -> None:
    base, ext = P._base_file(0), P._ext_file(0)
    print(f"base       {base.name:<26s} {'present' if base.exists() else 'MISSING'}"
          f"  ({P.MASTER_TOKENS:,} tokens, frozen)")
    if ext.exists():
        print(f"extension  {ext.name:<26s} present -- nothing to do")
        return
    print(f"extension  {ext.name:<26s} building {P.EXT_TOKENS:,} tokens ...")
    t0 = time.time()
    P.build_extension(0)
    mb = ext.stat().st_size / 1e6
    print(f"           done in {time.time() - t0:.0f} s, {mb:.0f} MB")

    s = P.get_stream(P.MASTER_TOKENS + 64_000)
    b = P.get_stream(P.MASTER_TOKENS)
    ok = (s.hi[:P.MASTER_TOKENS] == b.hi).all() and (s.y[:P.MASTER_TOKENS] == b.y).all()
    print(f"           base prefix unchanged: {ok};  total now {P.TOTAL_TOKENS:,} tokens")


if __name__ == "__main__":
    main()
