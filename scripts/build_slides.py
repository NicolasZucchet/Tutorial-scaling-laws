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
    uv run python scripts/build_slides.py --check    # placeholders, divs, steps
    uv run python scripts/build_slides.py --serve    # dev server (see below)

Setting COLLOQUIUM_GA_ID additionally injects a Google Analytics tag into the
built deck; see ANALYTICS_DOC.

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


DIV_GUARD_DOC = """A slide's `<div>`s have to balance.

colloquium wraps a slide's body in its own `<div class="slide-content">` and then
appends the footer inside the `<section>`.  A stray `</div>` in the slide source
closes that wrapper early, and colloquium's own closing `</div>` then has no div
of its own to close.  HTML's "have an element in scope" test does *not* treat
`<section>` as a barrier for a `</div>`, so the parser pops the section *and*
`<div class="colloquium-deck">` to find one -- and from that point on the deck is
over.  The offending slide's footer becomes a child of `<body>`, and so does
every slide after it.

What that looks like in a browser: two footers on screen at once (the orphaned
one, pinned to the bottom of the viewport, plus the current slide's own), and
every later slide laid out unscaled, ignoring the deck's 1280x720 transform.  It
is invisible in `colloquium capture`, which prints from the stacked print
stylesheet, and it does not show up as a Python error anywhere -- the markdown is
perfectly well-formed as far as the build is concerned.  Hence this check.

Counted per slide over the *expanded* source, figures included, and after HTML
comments are stripped (`<div>` inside a comment is prose, not markup).
"""

# colloquium splits a deck on `\n---\s*\n` and nothing else (parse.py), and
# drops the blocks that strip to nothing, so the numbering below is its
# numbering.  The generated reference slides come after all of these.
SLIDE_SPLIT = re.compile(r"\n---\s*\n")
COMMENT = re.compile(r"<!--.*?-->", re.S)
DIV_OPEN = re.compile(r"<div\b")
DIV_CLOSE = re.compile(r"</div>")


def unbalanced_divs(text: str) -> list[tuple[int, str, int]]:
    """Slides whose `<div>`/`</div>` counts differ, over the expanded source.

    Returns one (slide number, title, opens - closes) per offending slide.  The
    slide number counts the same `---`-separated blocks colloquium turns into
    slides, so it matches the deck's own numbering.
    """
    # The first block is the frontmatter, which colloquium consumes before it
    # splits; slide 1 is the one after it.
    blocks = SLIDE_SPLIT.split(text)[1:]
    bad, number = [], 0
    for block in blocks:
        if not block.strip():
            continue
        number += 1
        body = COMMENT.sub("", expand(block))
        delta = len(DIV_OPEN.findall(body)) - len(DIV_CLOSE.findall(body))
        if delta:
            title = re.search(r"(?m)^#+ (.*)$", block)
            bad.append((number, title.group(1).strip() if title else "(no heading)",
                        delta))
    return bad


def report_divs(text: str) -> int:
    """Print the div-balance report.  Returns the number of offending slides."""
    bad = unbalanced_divs(text)
    for number, title, delta in bad:
        kind = ("an unclosed <div>" if delta > 0 else "a stray </div>")
        print(f"  UNBALANCED   slide {number}: {abs(delta)} x {kind} -- "
              f"this breaks the deck from here on  ({title})", file=sys.stderr)
    return len(bad)

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


ANALYTICS_DOC = """Google Analytics, if COLLOQUIUM_GA_ID is set.

colloquium 0.2.3 has no hook for it.  Its `<head>` is a fixed template with five
substitution points -- title, font link, theme css, font css, `custom_css` --
and the frontmatter keys it reads are a closed list (parse.py), so a
`custom_head:` next to `custom_css:` is silently discarded.  The two things that
do reach the page are `custom_css`, inlined unescaped into `<style>`, and raw
HTML in a slide body.  Closing the `<style>` element early from inside
`custom_css` to smuggle a `<script>` into the head works and is a trap for
whoever reads the frontmatter next.  A `<script>` in the title slide's body also
works -- it is how assets/slides.css and assets/references.js get in -- but it
puts the tag inside a `<section>`, and colloquium base64s each slide's source
into a `data-colloquium-md` attribute, so the measurement ID would be in the
file twice.

So it goes in here, where the build already lives, as one insertion before
`</head>`.

The deck's own measurement ID is the default below, not an environment variable
you have to remember: a GA4 measurement ID is not a credential -- it ships in
the clear in the `<head>` of every page that uses it, and anything that could be
done with it can be done by reading the deck's own source.  What it does buy is
that `uv run python scripts/build_slides.py` produces the deck that is actually
hosted, with no second step that can be forgotten.

The environment still wins, for the two cases where it should:

    COLLOQUIUM_GA_ID=G-OTHER uv run python scripts/build_slides.py   # another property
    COLLOQUIUM_GA_ID= uv run python scripts/build_slides.py          # no tag at all

`--serve` deliberately does *not* inject it.  That path hands the expanded deck
to `colloquium serve`, which builds it itself, so the tag never gets added --
and it should not: the dev server is the one origin that really is http, so it
is the one place a tag would quietly fill the property with your own reloads.
Analytics belongs to the built artefact, which is the thing that gets hosted.

Note gtag only reports from an http(s) origin, so a deck opened over file:// is
unaffected either way -- this is for the hosted copy.
"""

GA_ENV = "COLLOQUIUM_GA_ID"
# This deck's own Google Analytics property.  See ANALYTICS_DOC for why it is a
# literal here and not a secret.
GA_ID_DEFAULT = "G-9DE3SJYSNF"
# A measurement ID, and nothing that could close the script element it lands in.
GA_ID = re.compile(r"\A[A-Za-z0-9-]{1,32}\Z")


def analytics_tag(ga_id: str) -> str:
    """The standard gtag.js snippet.  See ANALYTICS_DOC."""
    return (
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}">'
        f"</script>\n"
        f"<script>\n"
        f"  window.dataLayer = window.dataLayer || [];\n"
        f"  function gtag(){{dataLayer.push(arguments);}}\n"
        f"  gtag('js', new Date());\n"
        f"  gtag('config', '{ga_id}');\n"
        f"</script>\n"
    )


def analytics_id() -> str:
    """The configured measurement ID, or "" -- validated before anything is built.

    Checked up front rather than at insertion time: the insertion happens after
    colloquium has run, so a rejected ID there would fail having already spent
    the build, and leave its half-finished output behind.
    """
    # Unset falls back to the deck's own property; set-but-empty is how you say
    # "no tag", which an unset-means-default scheme would otherwise make
    # impossible to express.
    ga_id = os.environ.get(GA_ENV, GA_ID_DEFAULT).strip()
    if ga_id and not GA_ID.match(ga_id):
        raise SystemExit(f"{GA_ENV}={ga_id!r} is not a measurement ID "
                         f"(expected something like G-XXXXXXXXXX)")
    return ga_id


def with_analytics(html: str) -> str:
    """Insert the analytics tag before `</head>`, if an ID is in the environment."""
    ga_id = analytics_id()
    if not ga_id:
        return html
    if "</head>" not in html:
        raise SystemExit("built deck has no </head> to insert the analytics tag "
                         "before; colloquium's page template must have changed")
    return html.replace("</head>", analytics_tag(ga_id) + "</head>", 1)


def run(argv: list[str]) -> None:
    """Expand into BUILD_MD, hand it to colloquium, and clean up afterwards."""
    analytics_id()   # fail before spending the build, not after
    BUILD_MD.write_text(expand(SLIDES.read_text()))
    built = BUILD_MD.with_suffix(".html")
    try:
        subprocess.run(["colloquium", *argv, str(BUILD_MD)], check=True, cwd=ROOT)
        if built.exists():
            # colloquium names its output after its input; put it back where the
            # deck is expected to live.
            built.write_text(with_analytics(built.read_text()))
            built.replace(ROOT / "slides.html")
    finally:
        BUILD_MD.unlink(missing_ok=True)
        # Nothing half-built survives a failure: the intermediate is pid-tagged,
        # so a leftover would sit in the repository until someone noticed it.
        built.unlink(missing_ok=True)


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
        # Div balance is a source-level property, so it needs no build.  See
        # DIV_GUARD_DOC for what an imbalance does to the rendered deck.
        if report_divs(text):
            print("\nUnbalanced <div>s tear the deck apart from that slide on: "
                  "see above.", file=sys.stderr)
            status = 1
        else:
            print("  ok       every slide's <div>s balance")
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

    if report_divs(text):
        raise SystemExit("Refusing to build: see above.")
    run(["build"])
    n = len(keys(text))
    ga = analytics_id()
    print(f"Built: slides.html ({n} figures from figures/"
          + (f", analytics {ga}" if ga else ", no analytics") + ")")
    if report_steps((ROOT / "slides.html").read_text()):
        raise SystemExit("Refusing to call that a good build: see above.")


if __name__ == "__main__":
    main()
