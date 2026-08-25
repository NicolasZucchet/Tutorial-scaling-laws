"""Build the deck, resolving the `<!-- figure: key -->` placeholders in slides.md.

No plot or chart lives in slides.md: every figure sits in its own
`figures/<key>.md` fragment -- raw SVG for the hand-drawn ones, a colloquium
```chart fence for the plotted ones, each with whatever legend and
`<script src>` styles it -- and the slide carries only a one-line placeholder

    <!-- figure: zipf-fig -->

Some fragments are hand-edited (`zipf-fig`, `embed-fig`, `hebb-fig`,
`sphere-fig`, `loss-step-fig`, `scaling-twin-fig`, `pc-facts`,
`pc-bitstrings`, `finite-chart`); three are owned and overwritten by the script
that computes their numbers (`capacity-chart`, `isoflop-figure`,
`results-alpha`) and say so in a header comment.

colloquium has no include directive -- it reads nothing but its own theme files
at build time -- so the placeholders have to be expanded before it runs.  That
is this script's whole job:

    uv run python scripts/build_slides.py            # -> slides.html
    uv run python scripts/build_slides.py --check    # placeholders resolve?
    uv run python scripts/build_slides.py --serve    # dev server (see below)

The expansion goes to a temporary file next to slides.md, so `assets/` and
`figures/` resolve the same way they do for a plain build, and the temporary
file is removed afterwards.  `colloquium build slides.md` still runs, but every
figure comes out empty -- use this script instead.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides.md"
FIGURES = ROOT / "figures"
# Expanded intermediate: same directory as slides.md so relative hrefs match,
# and pid-tagged so two concurrent builds cannot rename each other's output
# over slides.html -- that failure mode renders as silently missing figures.
BUILD_MD = ROOT / f".slides.build.{os.getpid()}.md"

PLACEHOLDER = re.compile(r"^[ \t]*<!--[ \t]*figure:[ \t]*([A-Za-z0-9_-]+)[ \t]*-->[ \t]*$",
                         re.MULTILINE)


def figure(key: str) -> str:
    path = FIGURES / f"{key}.md"
    if not path.exists():
        raise SystemExit(f"slides.md references {key}, but {path.relative_to(ROOT)} "
                         f"does not exist")
    return path.read_text().strip("\n")


def keys(text: str) -> list[str]:
    return [m.group(1) for m in PLACEHOLDER.finditer(text)]


def expand(text: str) -> str:
    """Replace every `<!-- figure: key -->` line with the figure's markup."""
    return PLACEHOLDER.sub(lambda m: figure(m.group(1)), text)


def check(text: str) -> int:
    """Report placeholders that cannot be resolved and figures nothing uses."""
    used = keys(text)
    bad = [k for k in used if not (FIGURES / f"{k}.md").exists()]
    orphans = sorted(p.stem for p in FIGURES.glob("*.md") if p.stem not in used)

    for key in used:
        print(f"  {'MISSING' if key in bad else 'ok     '}  {key}")
    for key in orphans:
        print(f"  ORPHAN   {key}  (figures/{key}.md is never referenced)")

    if bad:
        print(f"\n{len(bad)} placeholder(s) cannot be resolved", file=sys.stderr)
        return 1
    if orphans:
        print(f"\n{len(orphans)} figure file(s) unused", file=sys.stderr)
        return 1
    return 0


def run(argv: list[str]) -> None:
    """Expand into BUILD_MD, hand it to colloquium, and clean up afterwards."""
    BUILD_MD.write_text(expand(SLIDES.read_text()))
    try:
        subprocess.run(["colloquium", *argv, str(BUILD_MD)], check=True, cwd=ROOT)
    finally:
        BUILD_MD.unlink(missing_ok=True)
        # colloquium names its output after its input; put it back where the
        # deck is expected to live.
        built = BUILD_MD.with_suffix(".html")
        if built.exists():
            built.replace(ROOT / "slides.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify every placeholder resolves, then exit")
    ap.add_argument("--serve", action="store_true",
                    help="run colloquium's dev server on the expanded deck; it "
                         "watches the expanded copy, so re-run after editing")
    args = ap.parse_args()

    text = SLIDES.read_text()
    if args.check:
        raise SystemExit(check(text))

    if args.serve:
        BUILD_MD.write_text(expand(text))
        try:
            subprocess.run(["colloquium", "serve", str(BUILD_MD)], check=True, cwd=ROOT)
        finally:
            BUILD_MD.unlink(missing_ok=True)
        return

    run(["build"])
    n = len(keys(text))
    print(f"Built: slides.html ({n} figures from figures/)")


if __name__ == "__main__":
    main()
