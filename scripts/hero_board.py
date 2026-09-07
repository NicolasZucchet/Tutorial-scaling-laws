"""Check (and prefill, and QR) the lab scoreboard's Google-Sheet plumbing.

The scoreboard slide reads the form's responses sheet in the browser -- see
assets/hero-board.js for the mechanism and figures/hero-board.md for the three
setup steps.  Everything about that setup can fail silently in front of a room:
a sheet that was never link-shared answers a sign-in page, a renamed question
stops matching a column, a "Publish to web" URL is up to five minutes stale.
This script does exactly what the slide does, from the terminal, and prints what
it found -- so the failure happens now instead of during the lab.

    uv run python scripts/hero_board.py --sheet "<responses-sheet share link>"
    uv run python scripts/hero_board.py --current            # check what is wired
    uv run python scripts/hero_board.py --current --watch    # ... every 10 s

`--sheet` is the one-step version: it turns a sheet link into its live-CSV URL,
writes that into figures/hero-board.md and immediately reads it back.  It writes
it BASE64'D, in `data-csv-b64`, because slides.html is published and a responses
sheet that is readable by link is only as private as its URL.  That is obfuscation
and not secrecy -- the browser's network tab still shows the request -- but it
keeps the URL out of the page source, the repo and their search indexes.  Pass
`--plain` for a sheet nobody minds sharing.  `--current` decodes whatever is wired
and checks it, so the URL need not be typed at a shell (or land in its history).

It also writes the QR the slide shows, from the form's own URL:

    uv run --with segno python scripts/hero_board.py --qr "<form url>"

which overwrites figures/hero-qr.md with an inline SVG -- no runtime dependency
and no third-party QR service, so the slide works on a dead venue network too.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import re
import urllib.parse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# The same four rules assets/hero-board.js matches headers with, in the same
# priority order.  Kept in step by hand: two implementations, one contract, and
# this one exists precisely to catch the case where a form's wording drifts out
# of that contract.
COLUMNS = [
    ("predicted", re.compile(r"predict|guess|expect", re.I)),
    ("actual", re.compile(r"actual|obtain|achiev|measur|real|got", re.I)),
    ("share", re.compile(r"%|percent|share|fraction|compute|flop|budget", re.I)),
]


def columns(header: list[str]) -> dict[str, int]:
    of, taken = {}, set()
    for role, rule in COLUMNS:
        for i, h in enumerate(header):
            if i not in taken and rule.search(h):
                of[role] = i
                taken.add(i)
                break
    return of


def number(cell: str) -> float | None:
    try:
        return float(re.sub(r"[,\s%]", "", cell))
    except (TypeError, ValueError):
        return None


def share(cell: str) -> float | None:
    x = number(cell)
    if x is None:
        return None
    return x if "%" in str(cell) else (x * 100 if x <= 1 else x)


def fetch(url: str) -> str:
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}_={int(time.time())}",
                                 headers={"User-Agent": "hero-board-check"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8-sig")


def rows(text: str) -> tuple[list[str], list[dict]]:
    table = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not table:
        raise SystemExit("the sheet is empty -- not even a header row")
    header = [h.strip() for h in table[0]]
    of = columns(header)
    missing = [k for k in ("predicted", "actual") if k not in of]
    if missing:
        raise SystemExit(
            f"no column matched {' and '.join(missing)}.  The sheet's headers are:\n"
            + "\n".join(f"  - {h}" for h in header)
            + "\nRename the form's questions so each contains its keyword "
              "(predicted / obtained / fraction), or edit COLUMNS in both this file "
              "and assets/hero-board.js.")
    out: list[dict] = []
    for r in table[1:]:
        def cell(role, row=r):
            j = of.get(role)
            return row[j] if j is not None and j < len(row) else ""
        p, a = number(cell("predicted")), number(cell("actual"))
        if p is None or a is None:
            continue                                  # a partial row is not a point
        # One dot per row, in sheet order: the form asks for no name, so there is
        # nothing to deduplicate on and a resubmission is a second point.
        out.append(dict(predicted=p, actual=a, share=share(cell("share"))))
    return header, out


def show(url: str) -> None:
    try:
        text = fetch(url)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(
                f"HTTP {e.code}: the sheet answered a sign-in page.  Open it, "
                "Share > General access > Anyone with the link > Viewer, and run this "
                "again -- nothing else needs changing.")
        raise SystemExit(f"could not fetch the CSV: {e}")
    except urllib.error.URLError as e:
        raise SystemExit(f"could not fetch the CSV: {e}")
    if text.lstrip().startswith("<"):
        raise SystemExit("that URL answered HTML, not CSV -- the sheet is probably "
                         "not shared with 'anyone with the link'.")
    header, data = rows(text)
    print(f"columns: {' | '.join(header)}")
    print(f"{len(data)} usable row(s)\n")
    if data:
        print(f"  {'#':<4}{'predicted':>10}{'obtained':>10}{'error':>9}{'share':>9}")
        for i, e in enumerate(data, 1):
            sh = "--" if e["share"] is None else f"{e['share']:.0f}%"
            print(f"  {i:<4}{e['predicted']:>10.4f}{e['actual']:>10.4f}"
                  f"{e['actual'] - e['predicted']:>+9.4f}{sh:>9}")
        err = sorted(e["actual"] - e["predicted"] for e in data)
        mid = err[len(err) // 2] if len(err) % 2 else (err[len(err) // 2 - 1]
                                                       + err[len(err) // 2]) / 2
        print(f"\nmedian error {mid:+.4f} nats -- the room predicts "
              f"{'too low' if mid > 0 else 'too high'}")


def qr(url: str) -> None:
    try:
        import segno
    except ImportError:
        raise SystemExit("needs segno: uv run --with segno python scripts/hero_board.py "
                         f"--qr '{url}'")
    # BytesIO, not StringIO: segno writes encoded bytes even for SVG.
    buf = io.BytesIO()
    # svgclass/omitsize so the deck's CSS owns the size, and a quiet zone of 2
    # modules because a QR flush to its container will not scan.
    segno.make(url, error="m").save(buf, kind="svg", xmldecl=False, svgns=True,
                                    border=2, omitsize=True, svgclass="hero-qr",
                                    lineclass="", dark="#0f3460", light=None)
    (FIGURES / "hero-qr.md").write_text(
        "<!-- GENERATED by scripts/hero_board.py --qr <form url>.  Do not hand-edit:\n"
        f"     re-run the script if the form moves.  Encodes:\n     {url} -->\n"
        '<div class="hero-qr-box">\n'
        + buf.getvalue().decode("utf-8").strip() + "\n</div>\n")
    print(f"-> figures/hero-qr.md  ({url})")


# A sheet URL as copied out of the browser, or a bare id.  Both the /d/<id>/ form
# of a normal sheet and the /d/e/<id>/ form of a published one are accepted, and
# the id is all that is kept.
_SHEET_ID = re.compile(r"/spreadsheets/d/(?:e/)?([A-Za-z0-9_-]{20,})")


# The tab, as the browser's own URL names it: ?gid=... or #gid=... .  Preferred
# over a tab *name* because it is what a copied link already carries, and because
# a name has to be spelled and localised right ("Form Responses 1" is only the
# English default).
_GID = re.compile(r"[?#&]gid=(\d+)")


def gviz(target: str, tab: str | None = None) -> str:
    """The live-CSV URL for a sheet, from its share link or its id.

    Which tab, in order of preference: an explicit `tab` name, else the `gid` in
    the pasted link, else nothing at all -- and with nothing, gviz serves the
    sheet's first visible tab.
    """
    m = _SHEET_ID.search(target)
    sheet_id = m.group(1) if m else target.strip().strip("/")
    gid = _GID.search(target)
    query = {"tqx": "out:csv"}
    if tab:
        query["sheet"] = tab
    elif gid:
        query["gid"] = gid.group(1)
    return (f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/gviz/tq?{urllib.parse.urlencode(query)}")


FRAGMENT = FIGURES / "hero-board.md"
_ATTR = re.compile(r'data-csv(?:-b64)?="([^"]*)"')


def unhide(value: str) -> str:
    """A URL, whether it arrives readable or base64'd."""
    if value.startswith("http"):
        return value
    try:
        return base64.b64decode(value + "===").decode("utf-8")
    except Exception:                                     # noqa: BLE001 - any of them
        raise SystemExit(f"{value[:24]}... is neither a URL nor base64")


def current() -> str:
    """The URL the slide is wired to, decoded."""
    m = _ATTR.search(FRAGMENT.read_text())
    if not m:
        raise SystemExit(f"no data-csv attribute in {FRAGMENT}")
    value = html.unescape(m.group(1)).strip()
    if not value or "PASTE" in value:
        raise SystemExit(f"{FRAGMENT} has no sheet wired yet -- run --sheet '<link>'")
    return unhide(value).split(",")[0]


def set_csv(url: str, plain: bool = False) -> None:
    """Point figures/hero-board.md at `url`, then read it the way the slide will.

    Base64 by default; see the module docstring for why, and for what that is and
    is not worth.  &amp; in the readable form: a bare & in an attribute is what a
    browser forgives rather than what the markup means, and this file is committed.
    """
    attr = ('data-csv="' + html.escape(url, quote=True) + '"' if plain else
            'data-csv-b64="' + base64.b64encode(url.encode()).decode() + '"')
    text, n = _ATTR.subn(attr, FRAGMENT.read_text(), count=1)
    if n != 1:
        raise SystemExit(f"expected a data-csv attribute in {FRAGMENT}, found {n}")
    print(f"-> {FRAGMENT.relative_to(ROOT)}  {attr.split('=')[0]}"
          f"{'' if plain else '  (the URL is not in the file, or in the deck, in readable form)'}\n"
          "   rebuild the deck with `uv run python scripts/build_slides.py`\n")
    FRAGMENT.write_text(text)
    show(url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="a responses-sheet CSV URL (or its base64) to read")
    ap.add_argument("--current", action="store_true",
                    help="read whatever figures/hero-board.md is wired to")
    ap.add_argument("--watch", action="store_true", help="re-read every 10 s")
    ap.add_argument("--plain", action="store_true",
                    help="with --sheet: write the URL readable, not base64'd")
    ap.add_argument("--qr", metavar="FORM_URL", help="write figures/hero-qr.md")
    ap.add_argument("--sheet", metavar="SHEET_URL_OR_ID",
                    help="point figures/hero-board.md at this responses sheet, "
                         "then check it")
    ap.add_argument("--tab", help="tab name, if not the sheet's first one")
    args = ap.parse_args()
    if args.qr:
        qr(args.qr)
    if args.sheet:
        set_csv(gviz(args.sheet, args.tab), plain=args.plain)
    url = current() if args.current else (unhide(args.csv) if args.csv else None)
    if url:
        while True:
            show(url)
            if not args.watch:
                break
            time.sleep(10)
            print("\n" + "-" * 60 + "\n")
    if not any((args.qr, args.csv, args.sheet, args.current)):
        ap.print_help(sys.stderr)


if __name__ == "__main__":
    main()
