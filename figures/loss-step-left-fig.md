<!-- The step function on its own, for the slide that only asserts it: zero loss
     below capacity, l above.  Nothing here but figures/loss-step-panel.md; the
     shaded block, the carry-over and the N^(1-alpha) law belong to
     loss-step-fig, two slides later, and this slide has no clicks.
     No blank line in this comment, or anywhere in this file: it is spliced into
     the slide right above an `<svg>`, and a blank line inside a raw-HTML run
     ends it -- after which these indented lines are parsed as an indented code
     block and a <pre> lands inside the figure.  See figures/loss-step-panel.md.
     The viewBox is the wide one shifted, not a tight crop: the panel is drawn at
     x = 92..530 of loss-step-fig's 1180-unit width, and `-279 0 1180 400` keeps
     that width -- so one viewBox unit is the same number of screen pixels, and
     the plot is the same size here as it is there -- while moving the origin left
     by 279 units, which centres 92..530 in the frame.  A viewBox of `60 0 500
     400` would centre it too, and blow the type up by 2.4x against the rest of
     the deck.
     This is 400 viewBox units at full slide width, i.e. a third of the slide's
     height once rendered, so the slide it sits on has room for the heading and
     ONE paragraph above it.  Add a second and colloquium drops the figure
     outright -- not clipped, not scaled: gone, on screen and in `capture`.  That
     is what this slide did for its first two commits. -->
<svg class="plot-fig" viewBox="-279 0 1180 400" role="img" aria-label="The per-context loss against the context index: zero for every context below the model's capacity, and l for every context above it.">
<!-- figure: loss-step-panel -->
</svg>
