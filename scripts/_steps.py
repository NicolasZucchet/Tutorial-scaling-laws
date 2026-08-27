"""Emitting animation steps that colloquium will actually count.

A slide's step count is not something the deck infers at run time.  colloquium
counts the `data-colloquium-fragment="1"` markers *it* generated and writes the
total into `data-fragment-count`; the presentation engine reads only that:

    this.fragmentCounts = this.slides.map(
        s => parseInt(s.getAttribute('data-fragment-count') || '0', 10)

A hand-written `data-fragment-index="k"` is passed through untouched and never
counted.  A figure whose reveals are all hand-indexed therefore lands on a slide
that reports zero steps: the right-arrow key skips to the next slide and none of
those elements is ever shown.  `colloquium capture` forces every fragment
visible, so the damage is invisible in a screenshot -- it only shows up when
someone actually presents.  `scripts/build_slides.py --check` guards against it.

Hand-written indices are still the right tool when several elements share one
step, or when a figure element has to land together with a prose beat.  They
just need marker-generated steps on the same slide to back them.  Hence this:

    step = Steps()
    ...
    f'<g class="fragment"{step.attr(1)}>'   # -> data-colloquium-fragment="1"
    f'<g class="fragment"{step.attr(1)}>'   # -> data-fragment-index="1"
    f'<g class="fragment"{step.attr(2)}>'   # -> data-colloquium-fragment="1"

The first element of each step carries the marker, later ones on the same step
carry the explicit index that marker is going to be given.  That only lines up
because colloquium numbers markers sequentially in document order, so steps have
to be *emitted* in ascending order with no gaps -- `attr` raises if they are not,
since the failure is otherwise silent and off-by-one.
"""

from __future__ import annotations


class Steps:
    """Per-figure reveal-attribute allocator.  One instance per figure file."""

    def __init__(self) -> None:
        self._opened: set[int] = set()

    def attr(self, step: int) -> str:
        """The reveal attribute for an element belonging to *step*.

        Includes its own leading space, so it drops straight into an f-string
        after `class="fragment"`.
        """
        if step < 1:
            raise ValueError(f"steps are 1-based; got {step}")
        if step in self._opened:
            return f' data-fragment-index="{step}"'
        if step != len(self._opened) + 1:
            raise ValueError(
                f"step {step} opened after {sorted(self._opened) or 'none'}: "
                "colloquium numbers markers in document order, so a figure has "
                "to open its steps in ascending order with no gaps, or every "
                "later index silently points at the wrong step"
            )
        self._opened.add(step)
        return ' data-colloquium-fragment="1"'

    @property
    def count(self) -> int:
        """How many steps this figure has opened."""
        return len(self._opened)
