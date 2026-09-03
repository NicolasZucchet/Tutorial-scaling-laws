<!-- Hand-written figure: no generator script.  Re-digitized 2026-08-30.

     Allen-Zhu & Li 2024, "Physics of Language Models: Part 3.3, Knowledge
     Capacity Scaling Laws", arXiv:2404.05405v1, FIGURE 1(a) -- the LEFT of the
     two panels, the 1000-EXPOSURE setting.  Its subcaption in the PDF reads
     "(a) bioS(N) data -- 1000 exposures -- peak R(F) >= 2"; the right panel is
     "(b) ... 100 exposures -- peak R(F) >= 1".  The panel matters: the dashed
     2 bits/param yardstick below is the paper's claim for 1000 exposures only
     (their Result 1(a)).  An earlier version of this file plotted panel (b) --
     where the paper's own result is the weaker R(F) >= 1 and the ratio peaks
     near 1.4-1.7 -- against the 1000-exposure line, which is why every point
     used to sit visibly below it; its x values were also uniformly ~7.4% too
     small.  Both are fixed here.  If you touch these numbers, say which panel
     they came from.

     HOW THE COORDINATES WERE RECOVERED.  The arXiv figures are vector
     graphics, so marker centres are exact and nothing was eyeballed:

       pdftocairo -svg -f 9 -l 9 2404.05405v1.pdf p9.svg

     In that SVG every matplotlib marker is a small circle path drawn in local
     coordinates under transform="matrix(0.312,0,0,-0.312,cx,cy)", so the
     translation (cx,cy) IS the marker centre in PostScript points.  Series are
     told apart by stroke colour and the colour -> N map read off the legend
     swatches (the legend markers are the ones drawn in panel-local coordinates
     under the panel's own transform, at x = 262.05, y = 200.83 down to 151.34).
     Axes are calibrated from the panel's own major ticks, not from the image
     border: x has 10^6 at 113.840 and 10^8 at 240.145 (63.1525 pt/decade),
     y has 10^6 at 191.750 and 10^9 at 82.156 (36.5313 pt/decade).  Each tick
     label's bounding-box centre lands within 0.8 pt of its tick.

     CALIBRATION CHECK 1 -- the 88M anchor.  The paper says "GPT2small ... is
     counted as having only 88M parameters in this paper".  The recovered
     model-size grid contains 87.57 M params.  PASS.

     CALIBRATION CHECK 2 -- the parameter-count formula.  For GPT2-l-h with the
     paper's reduced 3275-token vocabulary, params ~= 12*l*(64h)^2 + 3275*(64h):
       GPT2-2-2    formula 0.812   recovered 0.816   (+0.5%)
       GPT2-6-4    formula 5.557   recovered 5.577   (+0.4%)
       GPT2-6-6    formula 11.87   recovered 11.91   (+0.3%)
       GPT2-12-12  formula 87.45   recovered 87.57   (+0.1%)
     All within 0.5% (the formula omits biases and layernorms), and the l-h
     label the PDF prints beside each of those dots matches.  PASS.

     Three further checks, all passing:
       * under this calibration the panel's own "2 bit / param" guide line
         reads R = 2.000 at both of its endpoints;
       * panel (b) extracted the same way reproduces the independently known
         Figure 1(b) grid, 0.816 ... 611 M params, to four significant digits;
       * the three isolated large-N dots carry the labels 16-8, 6-20 and
         20-16, and the caption names GPT2-20-16 on bioS(10M) at 1000 exposures
         as its most expensive run -- GPT2-20-16 is 255 M params, exactly where
         the 10M dot sits.

     SERIES DRAWN: N = 50K, 100K, 200K, 500K, 1M, 2M facts, one dot per model
     the authors actually trained.  R below is bits stored per parameter, i.e.
     y/x, so the dashed line is R = 2:

       N      dots   x range (M params)   peak R   min R
       50K     29     0.816 -  19.0        1.93     0.15
       100K    33     1.213 -  33.2        2.16     0.17
       200K    33     2.409 -  63.23       2.35     0.18
       500K    35     5.577 - 175.0        2.72     0.16
       1M      36    11.91  - 356.0        2.71     0.15
       2M       1    52.12                 2.01     2.01

     Nothing is interpolated and nothing is extended to the left of a series'
     first dot: for each N the authors start at P ~= 11N, the size at which the
     model is just capacity-limited, so the series genuinely begin at different
     x.  Below that the paper predicts the 2-bit line but never measured it.
     N = 2M has exactly ONE model in the 1000-exposure panel (GPT2-16-8,
     52.1 M params) -- these runs are expensive; in panel (b) it has ten, which
     is another tell that the old numbers came from (b).

     SERIES NOT DRAWN, and why:
       * 10K and 20K peak at R = 0.74 and 1.32, entirely below the 2 bits/param
         line -- not a deficiency of the models but of the experiment grid: the
         smallest model trained here (0.816 M params) is already far larger than
         10-20K facts need, so these series are saturated everywhere and would
         read as counter-evidence to the very claim the panel makes.  They also
         buy no x-range: 50K starts at the same 0.816 M model.
       * 5M and 10M are single dots (122.3 M params, R = 2.05; 255.3 M params,
         R = 1.95).  2M is kept as the panel's high-N anchor because the deck
         has always named it; two further one-dot legend entries buy little.
       * Appendix Figure 9 repeats Figure 1 with 1-layer transformers and adds
         one model at 0.618 M params.  Excluded: the paper cautions that
         "1-layer transformers show a minor capacity ratio deficiency".

     COLOURS.  Six series need six ordered stops.  RAMP in
     scripts/chinchilla_svg.py is the deck's canonical single-hue ramp, but its
     six literal stops are not evenly spaced in lightness -- the first three are
     only 5.5 L* apart, which is too close to tell apart as categories.  These
     five of these six are that same ramp sampled through its own ramp_at() at
     the t values that make the steps even, t = 0, 0.367, 0.507, 0.640, 0.820:
     the hue path and the light endpoint are the deck's, and those steps come
     out ~10 L* each (against ~14.6 for the four stops figures/pc-bitstrings.md
     uses beside it, which is the price of six series over one lightness range).

     The sixth stop is NOT the ramp's own dark end.  The plateaus are ~2x apart
     in y, so position does most of the work of telling series apart and ~10 L*
     is enough -- except for 2M, which is a single dot sitting directly above
     the 1M plateau at nearly the same x, where position does no work at all
     and a reader can take it for a stray 1M point.  So the dark end is widened
     rather than the ramp re-ordered: the last stop is #0a2d56 (L* 18.4, still
     on the same hue path) instead of the ramp's #0d366c (23.0), which buys 14.3
     L* for the one step that has to carry the whole distinction.  The ladder is
     L* = 72.7, 62.6, 52.7, 42.8, 32.7, 18.4; steps 10.2, 9.9, 9.9, 10.1, 14.3.

     The five ramp-derived literals below are byte-for-byte what ramp_at() prints
     at those t against the ramp in scripts/_palette.py, so recomputing them lands
     on the same hexes rather than a pair that differs in one channel and reads as
     two colours in a grep.  Two of them used to be off by one unit that way.

     The six stops are also the ladder figures/pc-bitstrings.md draws its four
     series from, taking stops 1, 3, 5 and 6, so the two panels on the slide read
     as one colour scheme; see the COLOURS note there.

     LEGEND.  Seven keys have to fit the 744px half-column on one line, or the
     left plot drops below the right one and the captions stop being level.  So:
     no "N = " prefix (figures/pc-bitstrings.md has none either, and "2M facts"
     already carries the unit the way its "7M params" does), 11px marker
     swatches rather than 16, a 24px dash swatch rather than 30, and 0.6em
     between keys -- see the `.pc-slide .cap-legend` comment in
     assets/slides.css.  Measured: 706px of 744.  Adding a key will wrap it.

     AXIS.  x is in parameters, not in millions of them, so its ticks read
     1M / 10M / 100M -- the way every other model size in the deck is written --
     and the title is the deck's plain "model size N".  The y values stay in
     M bits, so the dashed guide is still y = 2x/1e6 and still reads R = 2.

     WHERE THE DATA SITS RELATIVE TO THE DASHED LINE.  Five of the six series
     reach or clear 2 bits/param at their peak; 50K comes within 4% (its best
     model, 1.01 M params, is already a little bigger than 50K facts need).
     Every series then falls away below the line to the right; that is
     saturation -- once the data is learned, extra parameters store nothing --
     and it is the shape the paper's figure has too, not a digitizing error. -->
<div class="cap-legend">
<span><svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true"><circle cx="5.5" cy="5" r="3" fill="#86b6ef"/></svg>50K</span>
<span><svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true"><circle cx="5.5" cy="5" r="3" fill="#599ae8"/></svg>100K</span>
<span><svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true"><circle cx="5.5" cy="5" r="3" fill="#3b7fd2"/></svg>200K</span>
<span><svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true"><circle cx="5.5" cy="5" r="3" fill="#2265b7"/></svg>500K</span>
<span><svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true"><circle cx="5.5" cy="5" r="3" fill="#174d91"/></svg>1M</span>
<span><svg width="11" height="10" viewBox="0 0 11 10" aria-hidden="true"><circle cx="5.5" cy="5" r="3" fill="#0a2d56"/></svg>2M facts</span>
<span><svg width="24" height="10" viewBox="0 0 24 10" aria-hidden="true"><line x1="1" y1="5" x2="23" y2="5" stroke="var(--fig-guide)" stroke-width="var(--fig-hair-width)" stroke-dasharray="6 5"/></svg>2 bits/param</span>
</div>

```chart
type: line
data:
  datasets:
    - label: "50K facts"
      color: "#86b6ef"
      data:
        - {x: 816000, y: 1.4}
        - {x: 1014000, y: 1.957}
        - {x: 1213000, y: 2.163}
        - {x: 1411000, y: 2.459}
        - {x: 1519000, y: 2.558}
        - {x: 1609000, y: 2.688}
        - {x: 1807000, y: 2.853}
        - {x: 2006000, y: 2.87}
        - {x: 2409000, y: 2.909}
        - {x: 2418000, y: 2.915}
        - {x: 2853000, y: 2.918}
        - {x: 3208000, y: 2.911}
        - {x: 3298000, y: 2.907}
        - {x: 3998000, y: 2.916}
        - {x: 4788000, y: 2.918}
        - {x: 4807000, y: 2.919}
        - {x: 5577000, y: 2.919}
        - {x: 6582000, y: 2.922}
        - {x: 7157000, y: 2.922}
        - {x: 7983000, y: 2.923}
        - {x: 8356000, y: 2.924}
        - {x: 8736000, y: 2.924}
        - {x: 10130000, y: 2.926}
        - {x: 10320000, y: 2.924}
        - {x: 11910000, y: 2.926}
        - {x: 13480000, y: 2.926}
        - {x: 15450000, y: 2.928}
        - {x: 17230000, y: 2.928}
        - {x: 19000000, y: 2.928}
    - label: "100K facts"
      color: "#599ae8"
      data:
        - {x: 1213000, y: 2.589}
        - {x: 1411000, y: 2.861}
        - {x: 1519000, y: 3.134}
        - {x: 1609000, y: 3.47}
        - {x: 1807000, y: 3.605}
        - {x: 2006000, y: 3.861}
        - {x: 2409000, y: 4.609}
        - {x: 2418000, y: 4.687}
        - {x: 2853000, y: 5.212}
        - {x: 3208000, y: 5.575}
        - {x: 3298000, y: 5.583}
        - {x: 3998000, y: 5.703}
        - {x: 4788000, y: 5.728}
        - {x: 4807000, y: 5.734}
        - {x: 5577000, y: 5.739}
        - {x: 6582000, y: 5.716}
        - {x: 7157000, y: 5.717}
        - {x: 7983000, y: 5.724}
        - {x: 8356000, y: 5.726}
        - {x: 8736000, y: 5.722}
        - {x: 10130000, y: 5.734}
        - {x: 10320000, y: 5.729}
        - {x: 11910000, y: 5.738}
        - {x: 13480000, y: 5.734}
        - {x: 15450000, y: 5.742}
        - {x: 17230000, y: 5.745}
        - {x: 19000000, y: 5.746}
        - {x: 20590000, y: 5.749}
        - {x: 22550000, y: 5.749}
        - {x: 23740000, y: 5.749}
        - {x: 26900000, y: 5.752}
        - {x: 30870000, y: 5.751}
        - {x: 33200000, y: 5.753}
    - label: "200K facts"
      color: "#3b7fd2"
      data:
        - {x: 2409000, y: 5.57}
        - {x: 2418000, y: 5.385}
        - {x: 2853000, y: 6.699}
        - {x: 3208000, y: 7.37}
        - {x: 3298000, y: 7.495}
        - {x: 3998000, y: 8.583}
        - {x: 4788000, y: 9.718}
        - {x: 4807000, y: 9.514}
        - {x: 5577000, y: 10.59}
        - {x: 6582000, y: 11.16}
        - {x: 7157000, y: 11.18}
        - {x: 7983000, y: 11.23}
        - {x: 8356000, y: 11.24}
        - {x: 8736000, y: 11.22}
        - {x: 10130000, y: 11.26}
        - {x: 10320000, y: 11.25}
        - {x: 11910000, y: 11.17}
        - {x: 13480000, y: 11.15}
        - {x: 15450000, y: 11.2}
        - {x: 17230000, y: 11.21}
        - {x: 19000000, y: 11.22}
        - {x: 20590000, y: 11.23}
        - {x: 22550000, y: 11.23}
        - {x: 23740000, y: 11.24}
        - {x: 26900000, y: 11.25}
        - {x: 30870000, y: 11.25}
        - {x: 33200000, y: 11.26}
        - {x: 39510000, y: 11.26}
        - {x: 43550000, y: 11.25}
        - {x: 45040000, y: 11.26}
        - {x: 52120000, y: 11.27}
        - {x: 59220000, y: 11.28}
        - {x: 63230000, y: 11.27}
    - label: "500K facts"
      color: "#2265b7"
      data:
        - {x: 5577000, y: 15.16}
        - {x: 6582000, y: 15.93}
        - {x: 7157000, y: 17.86}
        - {x: 7983000, y: 18.13}
        - {x: 8356000, y: 19.02}
        - {x: 8736000, y: 19.88}
        - {x: 10130000, y: 21.98}
        - {x: 10320000, y: 21.35}
        - {x: 11910000, y: 24.05}
        - {x: 13480000, y: 26.7}
        - {x: 15450000, y: 27.4}
        - {x: 17230000, y: 27.49}
        - {x: 19000000, y: 27.54}
        - {x: 20590000, y: 27.57}
        - {x: 22550000, y: 27.59}
        - {x: 23740000, y: 27.61}
        - {x: 26900000, y: 27.63}
        - {x: 30870000, y: 27.65}
        - {x: 33200000, y: 27.66}
        - {x: 39510000, y: 27.67}
        - {x: 43550000, y: 27.67}
        - {x: 45040000, y: 27.68}
        - {x: 52120000, y: 27.69}
        - {x: 59220000, y: 27.7}
        - {x: 63230000, y: 27.69}
        - {x: 78930000, y: 27.7}
        - {x: 82900000, y: 27.7}
        - {x: 87570000, y: 27.71}
        - {x: 104100000, y: 27.72}
        - {x: 115900000, y: 27.72}
        - {x: 122300000, y: 27.72}
        - {x: 144300000, y: 27.73}
        - {x: 154500000, y: 27.73}
        - {x: 161600000, y: 27.72}
        - {x: 175000000, y: 27.72}
    - label: "1M facts"
      color: "#174d91"
      data:
        - {x: 11910000, y: 32.26}
        - {x: 13480000, y: 35.38}
        - {x: 15450000, y: 37.09}
        - {x: 17230000, y: 40.25}
        - {x: 19000000, y: 42.24}
        - {x: 20590000, y: 44.82}
        - {x: 22550000, y: 48.78}
        - {x: 23740000, y: 51.05}
        - {x: 26900000, y: 52.96}
        - {x: 30870000, y: 53.94}
        - {x: 33200000, y: 54.05}
        - {x: 39510000, y: 54.19}
        - {x: 43550000, y: 54.2}
        - {x: 45040000, y: 54.25}
        - {x: 52120000, y: 54.31}
        - {x: 59220000, y: 54.33}
        - {x: 63230000, y: 54.34}
        - {x: 78930000, y: 54.39}
        - {x: 82900000, y: 54.39}
        - {x: 87570000, y: 54.41}
        - {x: 104100000, y: 54.44}
        - {x: 115900000, y: 54.46}
        - {x: 122300000, y: 54.45}
        - {x: 144300000, y: 54.47}
        - {x: 154500000, y: 54.47}
        - {x: 161600000, y: 54.47}
        - {x: 175000000, y: 54.53}
        - {x: 204900000, y: 54.49}
        - {x: 231700000, y: 54.5}
        - {x: 240300000, y: 54.49}
        - {x: 255300000, y: 54.49}
        - {x: 279700000, y: 54.5}
        - {x: 305700000, y: 54.49}
        - {x: 319000000, y: 54.49}
        - {x: 345000000, y: 54.5}
        - {x: 356000000, y: 54.48}
    - label: "2M facts"
      color: "#0a2d56"
      data:
        - {x: 52120000, y: 104.8}
    - label: "ref: 2 bits/param"
      color: "var(--fig-guide)"
      data:
        - {x: 600000, y: 1.2}
        - {x: 500000000, y: 1000}
options:
  plot:
    # Styling is declared here and read by assets/plot.js, the deck's one chart
    # layer; see the SCHEMA comment there for the vocabulary.  This panel and the
    # one beside it declare the same thing on purpose -- same line-plus-marker
    # treatment, same tick notation -- so the slide reads as one comparison and
    # not as two borrowed figures.
    #
    # Both panels get a line through their markers.  Allen-Zhu and Li publish
    # theirs as a bare scatter, and this used to reproduce that -- but next to
    # Morris's curves it read as a different kind of picture, and with ~30 dots
    # per series the eye has to join them anyway to see the plateau the panel is
    # about.  `markers: filled` keeps the joins straight (assets/plot.js gives
    # every series tension 0), so nothing is interpolated beyond "these dots are
    # one series".
    markers: filled
    guide: ["ref:"]
    # Both axes run over five decades, where "1,000,000" and "100,000,000" collide
    # and their commas are the only thing telling them apart.
    xTicks: si
    yTicks: si
  plugins:
    legend: {display: false}
  scales:
    x:
      type: logarithmic
      title: {display: true, text: "model size N"}
      min: 600000
      max: 500000000
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
    y:
      type: logarithmic
      title: {display: true, text: "memory stored (M bits)"}
      min: 1
      max: 1100
      grid: {drawOnChartArea: false}
      ticks: {padding: 8}
```

<script src="assets/plot.js"></script>
