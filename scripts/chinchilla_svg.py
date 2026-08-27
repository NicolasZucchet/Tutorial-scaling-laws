"""Panel-scoped SVG helpers for the three Chinchilla slides.

scripts/isoflop_slide.py draws the same kind of figure for the toy model, but its
`frame()` reads the plot box off module globals: one box, two panels side by side.
The Chinchilla slides need a wide main panel plus two stacked half-height panels, so
the geometry has to travel with the axis rather than with the module.  Everything else
-- the log axis, the ticks outside a hairline frame, the dashed fitted line with its
exponent set along it -- is the same construction and deliberately the same look.

Nothing here knows anything about Chinchilla; see scripts/chinchilla_slides.py.
"""

from __future__ import annotations

import numpy as np

# The deck's single-hue ramp, cheap -> expensive / small -> large.
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#256abf", "#184f95", "#0d366b")
# The slide background, for the halo a marker gets so it keeps its edge over a curve.
SURFACE = "#fcfcfb"


def num(v: float) -> str:
    """Axis numbers the way the deck writes them: 10k, 100k, 1M, 1.5."""
    for unit, sfx in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= unit:
            m = v / unit
            return f"{m:g}{sfx}"
    return f"{v:g}"


def sup(exp: str, tail: str = "", gap: float = 4.0) -> str:
    """A raised exponent, and whatever follows it put back on the baseline.

    `dy` shifts the text position, not the one tspan, so a trailing ")" stays raised
    unless it is shifted back; `gap` keeps a leading minus clear of an italic C.
    """
    out = f'<tspan dx="{gap:g}" dy="-9" font-size="0.72em">{exp}</tspan>'
    return out + (f'<tspan dy="9">{tail}</tspan>' if tail else "")


class Log:
    """A log10 axis mapping data to viewBox units, plus its power-of-ten ticks."""

    def __init__(self, lo: float, hi: float, p0: float, p1: float):
        self.lo, self.hi, self.p0, self.p1 = lo, hi, p0, p1

    def __call__(self, v: float) -> float:
        t = (np.log10(v) - np.log10(self.lo)) / (np.log10(self.hi) - np.log10(self.lo))
        return self.p0 + t * (self.p1 - self.p0)

    def decades(self) -> list[int]:
        k0 = int(np.ceil(np.log10(self.lo) - 1e-9))
        k1 = int(np.floor(np.log10(self.hi) + 1e-9))
        return list(range(k0, k1 + 1))


class Panel:
    """One plot box: two log scales, a hairline frame, and the drawing primitives.

    `x0 < x1` and `y1 < y0` in viewBox units (y grows downwards, so `y0` is the
    bottom of the box).  `clip` is the id of a clipPath covering the box, emitted by
    `defs()`; fitted lines run to the edge of the box and are clipped there.
    """

    def __init__(self, name: str, x0: float, x1: float, y0: float, y1: float,
                 xlim: tuple[float, float], ylim: tuple[float, float]):
        self.name, self.x0, self.x1, self.y0, self.y1 = name, x0, x1, y0, y1
        self.sx = Log(xlim[0], xlim[1], x0, x1)
        self.sy = Log(ylim[0], ylim[1], y0, y1)
        self.clip = f"cc-{name}"

    # -------------------------------------------------------------- structure

    def defs(self) -> str:
        return (f'<clipPath id="{self.clip}"><rect x="{self.x0}" y="{self.y1 - 10}" '
                f'width="{self.x1 - self.x0}" height="{self.y0 - self.y1 + 10}"/>'
                "</clipPath>")

    def axes(self, xlab: str | None, ylab: str, x_ticks, y_ticks,
             x_fmt=num, y_fmt=num, x_labels: bool = True,
             ylab_dx: float = 74, xlab_dy: float = 72) -> list[str]:
        """The two axes, ticks and labels outside them, and a rotated y title.

        `x_labels=False` draws the ticks but no numbers: the stacked right-hand
        panels share one compute axis, so only the lower one is annotated.
        """
        out = [f'<line class="pf-axis" x1="{self.x0}" y1="{self.y0}" x2="{self.x1}" '
               f'y2="{self.y0}"/>',
               f'<line class="pf-axis" x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" '
               f'y2="{self.y1}"/>']
        for v in x_ticks:
            x = self.sx(v)
            out.append(f'<line class="pf-axis" x1="{x:.0f}" y1="{self.y0}" '
                       f'x2="{x:.0f}" y2="{self.y0 + 8}"/>')
            if x_labels:
                out.append(f'<text class="pf-muted pf-small" x="{x:.0f}" '
                           f'y="{self.y0 + 34}" text-anchor="middle">{x_fmt(v)}</text>')
        for v in y_ticks:
            y = self.sy(v)
            out.append(f'<line class="pf-axis" x1="{self.x0 - 8}" y1="{y:.0f}" '
                       f'x2="{self.x0}" y2="{y:.0f}"/>')
            out.append(f'<text class="pf-muted pf-small" x="{self.x0 - 14}" '
                       f'y="{y + 7:.0f}" text-anchor="end">{y_fmt(v)}</text>')
        if xlab is not None:
            out.append(f'<text class="pf-muted pf-small" '
                       f'x="{(self.x0 + self.x1) / 2:.0f}" '
                       f'y="{self.y0 + xlab_dy:.0f}" text-anchor="middle">{xlab}</text>')
        cy = (self.y0 + self.y1) / 2
        out.append(f'<text class="pf-muted pf-small" transform="rotate(-90 '
                   f'{self.x0 - ylab_dx:.0f} {cy:.0f})" x="{self.x0 - ylab_dx:.0f}" '
                   f'y="{cy:.0f}" text-anchor="middle">{ylab}</text>')
        return out

    # -------------------------------------------------------------- marks

    def at(self, x: float, y: float) -> tuple[float, float]:
        return self.sx(x), self.sy(y)

    def path(self, pts) -> str:
        return "M " + " L ".join(f"{self.sx(x):.1f} {self.sy(y):.1f}" for x, y in pts)

    def curve(self, pts, colour: str, width: float = 2.4, cls: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        return (f'<path{c} d="{self.path(pts)}" fill="none" stroke="{colour}" '
                f'stroke-width="{width}" stroke-linecap="round" '
                'stroke-linejoin="round"/>')

    def dot(self, x: float, y: float, colour: str, r: float = 3.6,
            cls: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        return (f'<circle{c} cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" '
                f'fill="{colour}"/>')

    def diamond(self, x: float, y: float, colour: str, r: float = 6.0,
                cls: str = "") -> str:
        cx, cy = self.sx(x), self.sy(y)
        d = (f"M {cx:.1f} {cy - r:.1f} L {cx + r:.1f} {cy:.1f} L {cx:.1f} "
             f"{cy + r:.1f} L {cx - r:.1f} {cy:.1f} Z")
        c = f' class="{cls}"' if cls else ""
        return (f'<path{c} d="{d}" fill="{colour}" stroke="{SURFACE}" '
                'stroke-width="1.6"/>')

    def clipped(self, inner: list[str]) -> list[str]:
        return [f'<g clip-path="url(#{self.clip})">', *inner, "</g>"]

    # -------------------------------------------------------------- labels

    def label_on(self, p0, p1, t: float, base: str, exp: str, off: float = 15.0,
                 cls: str = "pf-muted pf-small") -> str:
        """An exponent label laid along the fitted line p0 -> p1 and lifted clear.

        Set at the fraction `t` of the way along the line, rotated to its angle and
        pushed `off` units up its normal, so the exponent reads as the line's slope
        rather than as a caption floating near it.  Both points are in viewBox units.
        """
        (x0, y0), (x1, y1) = p0, p1
        dx, dy = x1 - x0, y1 - y0
        n = float(np.hypot(dx, dy)) or 1.0
        x, y = x0 + t * dx + off * dy / n, y0 + t * dy - off * dx / n
        turn = float(np.degrees(np.arctan2(dy, dx)))
        return (f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" '
                f'transform="rotate({turn:.1f} {x:.1f} {y:.1f})" '
                f'text-anchor="middle">{base}{sup(exp)}</text>')

    def power_line(self, pref: float, exp: float, lo: float, hi: float,
                   n: int = 3) -> list[tuple[float, float]]:
        return [(x, pref * x**exp) for x in np.geomspace(lo, hi, n)]
