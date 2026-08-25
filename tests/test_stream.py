"""Guards on the training stream: the canonical prefix must never move.

`data.sample_tokens` sizes its rejection chunk from the requested length, so a longer
draw is *not* an extension of a shorter one.  Every recorded loss in this repo is
therefore tied to the exact bytes of `stream_master_s0.npz`, and these tests fail
loudly if anything regenerates it.

    PYTHONPATH=src uv run python tests/test_stream.py     # ~10 s (no extension needed)
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assocmem import problem as P  # noqa: E402
from assocmem import data as D  # noqa: E402


def test_base_fingerprint():
    """The canonical prefix is the instance every recorded result was measured on."""
    s = P.get_stream(4096)
    for name, want in P.BASE_FINGERPRINT.items():
        got = tuple(int(v) for v in getattr(s, name)[:3])
        assert got == want, (f"stream_master_s0.npz has been regenerated: {name}[:3] is "
                            f"{got}, expected {want}.  Every loss in REPORT.md and "
                            f"results/ledger.jsonl was measured on the old bytes.")
    print(f"  base fingerprint intact                      {P.BASE_FINGERPRINT}")


def test_sampler_is_not_nested():
    """Documents *why* the base file is frozen rather than regenerated on demand."""
    a = D.sample_tokens(1000, seed=1000)
    b = D.sample_tokens(100_000, seed=1000)
    assert not np.array_equal(a, b[:1000]), (
        "sample_tokens has become prefix-nested in m -- the two-file scheme in "
        "problem.py could now be simplified, but check every cached stream first.")
    print("  sampler is not prefix-nested in m            (hence the frozen base file)")


def test_prefixes_nested_within_base():
    """Runs at different step counts must see the same data, in the same order."""
    long = P.get_stream(200_000)
    for m in (4096, 64_000, 199_936):
        short = P.get_stream(m)
        assert np.array_equal(short.hi, long.hi[:m])
        assert np.array_equal(short.lo, long.lo[:m])
        assert np.array_equal(short.y, long.y[:m])
    print("  prefixes nested below MASTER_TOKENS          4096 / 64k / 200k")


def test_extension_is_appended_not_merged():
    """Beyond MASTER_TOKENS the base bytes must still come first, unchanged."""
    if not P._ext_file(0).exists():
        print("  extension not built -- skipping append check "
              "(run scripts/build_extension.py)")
        return
    base = P.get_stream(P.MASTER_TOKENS)
    long = P.get_stream(P.MASTER_TOKENS + 64_000)
    assert np.array_equal(long.hi[:P.MASTER_TOKENS], base.hi)
    assert np.array_equal(long.y[:P.MASTER_TOKENS], base.y)
    assert len(long.y) == P.MASTER_TOKENS + 64_000
    print("  extension appends, base prefix unchanged     "
          f"{P.MASTER_TOKENS:,} + 64,000")


def test_overlong_request_refused():
    try:
        P.get_stream(P.TOTAL_TOKENS + 1)
    except ValueError as e:
        assert "must not be touched" in str(e)
        print(f"  refuses > TOTAL_TOKENS                       {P.TOTAL_TOKENS:,}")
        return
    raise AssertionError("get_stream accepted more tokens than it holds")


def test_build_extension_refuses_to_clobber():
    if not P._ext_file(0).exists():
        print("  extension not built -- skipping clobber check")
        return
    try:
        P.build_extension(0, overwrite=True)
    except FileExistsError:
        print("  build_extension(overwrite=True) refuses      (delete by hand)")
        return
    raise AssertionError("build_extension overwrote an existing extension")


if __name__ == "__main__":
    print("stream guards")
    for f in (test_base_fingerprint, test_sampler_is_not_nested,
              test_prefixes_nested_within_base, test_extension_is_appended_not_merged,
              test_overlong_request_refused, test_build_extension_refuses_to_clobber):
        f()
    print("ok")
