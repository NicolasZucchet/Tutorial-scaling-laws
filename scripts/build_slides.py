"""Build the deck, resolving the `<!-- figure: key -->` placeholders in slides.md.

No plot or chart lives in slides.md: every figure sits in its own
`figures/<key>.md` fragment -- raw SVG for the hand-drawn ones, a colloquium
```chart fence for the plotted ones, each with whatever legend and
`<script src>` styles it -- and the slide carries only a one-line placeholder

    <!-- figure: zipf-fig -->

Some fragments are hand-edited (`zipf-fig`, `embed-fig`, `w-build-fig`,
`sphere-fig`, `loss-step-fig`, `scaling-twin-fig`, `pc-facts`,
`pc-bitstrings`, `finite-chart`); four are owned and overwritten by the script
that computes their numbers (`capacity-chart`, `isoflop-figure`,
`results-alpha`, `emergence-chart`) and say so in a header comment.

colloquium has no include directive -- it reads nothing but its own theme files
at build time -- so the placeholders have to be expanded before it runs.  That
is this script's whole job:

    uv run python scripts/build_slides.py            # -> slides.html
    uv run python scripts/build_slides.py --check    # placeholders resolve?
    uv run python scripts/build_slides.py --serve    # dev server (see below)

For a static build, the expansion goes to a temporary file next to slides.md
and is moved back to slides.html. In serve mode, an isolated live directory
keeps an expanded slides.md synchronized with the authored slides, figures,
bibliography, and assets, while exposing the stable `/slides.html` URL.
`colloquium build slides.md` still runs, but every figure comes out empty --
use this script instead.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

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


STEP_GUARD_DOC = """Every animation step has to be *reachable*.

colloquium counts a slide's steps by counting the markers it generated itself --
`<!-- step -->` in prose, or `data-colloquium-fragment="1"` in a figure -- and
writes the total into `data-fragment-count`.  The presentation engine reads only
that attribute:

    this.fragmentCounts = this.slides.map(
        s => parseInt(s.getAttribute('data-fragment-count') || '0', 10)

A hand-written `data-fragment-index="k"` is passed through untouched and *not*
counted.  So a slide whose reveals are all hand-indexed reports zero steps: the
right-arrow key skips straight to the next slide and every one of those elements
stays invisible for the whole talk.  That is not visible in `colloquium capture`,
which forces all fragments on, so it has to be checked here.

The rule this enforces: on every slide, `data-fragment-count` must be at least
the largest hand-written `data-fragment-index`.  Explicit indices are still the
right tool for syncing a figure element to a prose beat -- they just need at
least that many marker-generated steps on the same slide to back them.
"""

SECTION = re.compile(r'<section class="slide[^"]*"([^>]*)>(.*?)</section>', re.S)
FRAG_COUNT = re.compile(r'data-fragment-count="(\d+)"')
FRAG_INDEX = re.compile(r'data-fragment-index="(\d+)"')
SLIDE_INDEX = re.compile(r'data-index="(\d+)"')
HEADING = re.compile(r"<h[12][^>]*>(.*?)</h[12]>", re.S)
TAG = re.compile(r"<[^>]+>")


def unreachable_steps(html: str) -> list[tuple[int, str, int, int]]:
    """Slides whose hand-indexed reveals outnumber the steps colloquium counted.

    Returns one (slide number, title, declared count, highest index) per
    offending slide.
    """
    bad = []
    for attrs, body in SECTION.findall(html):
        index = SLIDE_INDEX.search(attrs)
        if not index:
            continue
        number = int(index.group(1)) + 1
        declared = FRAG_COUNT.search(attrs)
        declared = int(declared.group(1)) if declared else 0
        indices = [int(i) for i in FRAG_INDEX.findall(body)]
        if not indices:
            continue
        highest = max(indices)
        if highest > declared:
            title = HEADING.search(body)
            title = TAG.sub("", title.group(1)).strip() if title else "(no heading)"
            bad.append((number, title, declared, highest))
    return bad


def report_steps(html: str) -> int:
    """Print the unreachable-step report.  Returns the number of bad slides."""
    bad = unreachable_steps(html)
    for number, title, declared, highest in bad:
        print(f"  UNREACHABLE  slide {number}: declares {declared} step(s) but "
              f"hand-indexes up to {highest} -- "
              f"{highest - declared} reveal(s) can never be shown  ({title})",
              file=sys.stderr)
    return len(bad)


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


def build_to_string() -> str:
    """Build the deck and return the HTML without touching slides.html.

    `--check` needs to inspect colloquium's output -- step counts only exist
    after it has run -- but must not overwrite the committed build.
    """
    BUILD_MD.write_text(expand(SLIDES.read_text()))
    built = BUILD_MD.with_suffix(".html")
    try:
        subprocess.run(["colloquium", "build", str(BUILD_MD)], check=True, cwd=ROOT,
                       capture_output=True)
        return built.read_text()
    finally:
        BUILD_MD.unlink(missing_ok=True)
        built.unlink(missing_ok=True)


def watch_paths() -> list[pathlib.Path]:
    """Files whose changes should trigger a fresh placeholder expansion."""
    paths = [SLIDES, ROOT / "refs.bib"]
    paths.extend(sorted(FIGURES.glob("*.md")))
    paths.extend(sorted(p for p in (ROOT / "assets").rglob("*") if p.is_file()))
    return paths


def signature(paths: list[pathlib.Path]) -> tuple[tuple[str, int, int], ...]:
    """Cheap polling signature that notices edits, additions, and removals."""
    result = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        result.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(result)


def serve(port: int) -> None:
    """Serve an expanded deck and keep it synchronized with its sources."""
    # Colloquium names its output after its input. An isolated input named
    # `slides.md` gives us the stable `/slides.html` route without overwriting
    # the authored slides.md in the repository.
    with tempfile.TemporaryDirectory(prefix=".slides.live.", dir=ROOT) as tmp:
        live_root = pathlib.Path(tmp)
        live_md = live_root / "slides.md"

        # Keep all browser and bibliography-relative paths valid inside the
        # isolated serving directory while continuing to serve the real files.
        for name in ("assets", "fonts", "refs.bib"):
            source = ROOT / name
            if source.exists():
                (live_root / name).symlink_to(source, target_is_directory=source.is_dir())

        live_md.write_text(expand(SLIDES.read_text()))
        watched = watch_paths()
        previous = signature(watched)

        process = subprocess.Popen(
            ["colloquium", "serve", "-p", str(port), "slides.md"],
            cwd=live_root,
        )
        interrupted = False
        try:
            while process.poll() is None:
                time.sleep(0.5)
                current_paths = watch_paths()
                current = signature(current_paths)
                if current == previous:
                    continue
                live_md.write_text(expand(SLIDES.read_text()))
                watched = current_paths
                previous = current
                print("Sources changed; refreshed expanded deck.", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            process.terminate()
        finally:
            if process.poll() is None:
                process.terminate()
            returncode = process.wait()

        if returncode and not interrupted:
            raise subprocess.CalledProcessError(returncode, process.args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify every placeholder resolves, then exit")
    ap.add_argument("--serve", action="store_true",
                    help="serve the expanded deck and rebuild it when slides, "
                         "figures, bibliography, or assets change")
    ap.add_argument("-p", "--port", type=int, default=8090,
                    help="port for --serve (default: 8090)")
    args = ap.parse_args()

    text = SLIDES.read_text()
    if args.check:
        status = check(text)
        # Step reachability can only be read off colloquium's output, so this
        # half of the check needs a build of its own.  See STEP_GUARD_DOC.
        if report_steps(build_to_string()):
            print("\nSome animation steps are unreachable and will never show "
                  "during the talk.", file=sys.stderr)
            status = 1
        else:
            print("  ok       every animation step is reachable")
        raise SystemExit(status)

    if args.serve:
        serve(args.port)
        return

    run(["build"])
    n = len(keys(text))
    print(f"Built: slides.html ({n} figures from figures/)")
    if report_steps((ROOT / "slides.html").read_text()):
        raise SystemExit("Refusing to call that a good build: see above.")


if __name__ == "__main__":
    main()
