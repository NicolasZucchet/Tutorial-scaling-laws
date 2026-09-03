// The deck's one Chart.js layer, for all six ```chart fences.
//
// colloquium's chart element (elements/chart.py) can only carry `label`, `data`
// and `color` per series -- it drops every other per-dataset key -- and it ships
// its config to the browser as JSON in a `data-chart-config` attribute, so
// nothing that needs a *function* (a tick formatter, a hand-drawn label) can be
// written in the YAML at all.  What it does pass through untouched is any custom
// key under `options:`, and that is the whole design here: each fence declares a
// `plot:` block saying what kind of series and what kind of axis it wants, and
// this file is the one thing that reads it.  See SCHEMA below.
//
// It replaces assets/{capacity,results,emergence,pc,finite}-chart.js.  Those five
// had grown the same three routines three times over -- the "is this value a power
// of N" test, the gap from an axis line to its labels, and the ~30 lines that draw
// a mantissa with a raised exponent -- plus a fourth number formatter, and six
// different marker radii for one job.  Worse, each identified "its" chart by
// sniffing a dataset label or an axis title (`indexOf("parameter")`, `y.text ===
// "top-1 accuracy"`), which was never a handle so much as a coincidence that had
// not broken yet.  The `plot:` block is the handle, so the sniffing is gone.
//
// THE SCRIPT TAG'S POSITION NO LONGER MATTERS.  Three of the replaced files noted
// that their `<script src>` had to sit *after* the chart block, because the marker
// pass ran at parse time and could only see the canvases parsed so far; one of them
// depended on being loaded twice for exactly that reason.  That constraint is
// superseded rather than preserved: every pass here runs from a Chart.js plugin
// hook, so it runs once per chart when colloquium constructs it (in its `window`
// load handler, well after every canvas exists) and never has to find them itself.
// The cost is that six figures carrying six tags for one URL means six executions,
// so the whole file is guarded to do its work once.
(function () {
  if (window.__deckPlot) return;
  window.__deckPlot = true;

  /* ------------------------------------------------------------------ SCHEMA

     The whole vocabulary a ```chart fence can use, and the deck's reference
     for how a chart is styled.  Everything under `plot:` is optional; a fence
     that declares nothing gets Chart.js's own behaviour and the deck's type.

     options:
       plot:
         # ---- series ------------------------------------------------------------
         # Every name below matches a series if it appears anywhere in that series'
         # label, case-insensitively -- the same test the five old files made with
         # indexOf, and the reason a rule can say "theory" once and catch both
         # "theory, model axis" and "theory, data axis".
         markers: filled          # filled | hollow | none, for every series
         markerFor:               # per-series override of the above
           "measured lower bound": hollow
         # A series' own 'color:' (outside 'plot:', where colloquium reads it) may be
         # written as var(--deck-navy) rather than as a hex; see resolveColors.
         dash: ["first 10k, tested on all"]   # dashed, but still data
         guide: ["one memory per parameter"]  # a yardstick: dashed, thin, unmarked, behind
         scatter: ["measured"]    # markers only -- a line here would claim a model
         # ---- axes --------------------------------------------------------------
         # si       10k / 100k / 1M, on the decades inside the axis range
         # pow10    10 with a raised exponent, on the decades
         # pow2     2 with a raised exponent, on the powers of two
         # step:<n> a fixed tick step, e.g. step:0.25 -> 0, 0.25, 0.5, 0.75, 1
         # at:<v>,<v>,...  exactly these ticks, written as plain numbers
         # absent   whatever Chart.js would have done
         xTicks: si
         yTicks: step:0.25
         # ---- in-plot series labels --------------------------------------------
         # Chart.js has no text mark, so a chart that names its curves in the plot
         # instead of in a legend lists the notes here.  Each is anchored to a point
         # on the curve it names, nudged clear of it in pixels, and drawn in that
         # series' own colour.  _underscores_ mark italic maths variables, the same
         # split the SVG figures make with .fig-var.
         notes:
           - {series: "theory, model axis", at: [1.44, 0.44], nudge: [-8, -18],
              align: right, text: "_α_ − 1"}
  */

  // ---------------------------------------------------------------- tokens
  //
  // Chart.js defaults to 12px Helvetica and knows nothing about the deck, which
  // is why all six charts used to render their axis text at half the size of the
  // hand-drawn SVG figures beside them, in a different typeface.  Everything
  // below comes out of the token block in assets/slides.css, so the charts and
  // the figures are one family and a change of mind is still one edit.
  var INK = null;

  // Why the axis size is --fig-small and nothing else.  That token (20px) is what
  // the SVG figures set their tick labels and axis titles in, and the note at the
  // top of the token block explains that a figure's px are canvas units divided by
  // its own --fig-k.  A chart has no such factor: pinSize below gives the canvas
  // the same pixel size as the box it sits in, so a Chart.js px *is* a deck px and
  // the token applies unscaled.  The tallest of these charts is 380px in a
  // half-slide column, and at 20px nothing crowds -- the capacity chart's seven
  // powers-of-two ticks are the densest axis in the set and still hold a clear gap.
  // Measured against figures/kaplan-compute-fig.md's axis labels in the same
  // render, the chart labels come out within a pixel of them; checked on slides 35,
  // 36, 44, 45 and 46, on screen and in a PDF capture.

  // Marker radii.  --fig-point (2.8) is what every legend swatch in the deck is
  // drawn at, so a series whose line already carries it takes exactly that and the
  // key matches the curve.  Two roles need more ink and get it as a multiple of the
  // same token rather than as numbers of their own: a scatter has only its dots to
  // be seen by, and a ring reads smaller than a disc of equal radius.
  var SCATTER_K = 1.15;
  var HOLLOW_K = 1.45;

  function readTokens() {
    var css = getComputedStyle(document.documentElement);
    function tok(name) { return css.getPropertyValue(name).trim(); }
    return {
      family: tok("--colloquium-font-body"),
      muted: tok("--colloquium-muted"),
      surface: tok("--fig-surface") || "#ffffff",
      data: parseFloat(tok("--fig-data-width")),
      hair: parseFloat(tok("--fig-hair-width")),
      point: parseFloat(tok("--fig-point")),
      axisSize: parseFloat(tok("--fig-small")),
      noteSize: parseFloat(tok("--fig-text")),
      exp: parseFloat(tok("--fig-exp")),
      // The stylesheet names the italic-maths face in a rule rather than in a
      // token (.fig .fig-var), and a canvas cannot read a rule, so this asks for
      // a token that may not exist yet and falls back to the same three families.
      mathFamily: tok("--fig-math-family") ||
        '"Latin Modern Math", "Cambria Math", Georgia, serif',
      mathScale: parseFloat(tok("--fig-math")),
      dash: tok("--fig-dash").split(/[\s,]+/).map(Number),
    };
  }

  function wireDefaults() {
    INK = readTokens();
    Chart.defaults.font.family = INK.family;
    Chart.defaults.font.size = INK.axisSize;
    Chart.defaults.color = INK.muted;
    // colloquium writes borderWidth and pointRadius onto every line dataset
    // itself, so these two never win on a deck chart; they are set anyway so that
    // "the deck's line" has one definition, and the series pass below restates
    // them per dataset because it has to.
    Chart.defaults.elements.line.borderWidth = INK.data;
    Chart.defaults.elements.point.radius = INK.point;
  }

  // ---------------------------------------------------------------- helpers
  function spec(chart) {
    return ((chart.config.options || {}).plot) || null;
  }

  function hit(label, name) {
    return String(label || "").toLowerCase().indexOf(String(name).toLowerCase()) !== -1;
  }

  function listed(label, names) {
    return (names || []).some(function (n) { return hit(label, n); });
  }

  // k with base^k == v, or null.  Was written three times, twice hard-coded to 10.
  function exponent(base, v) {
    var k = Math.round(Math.log(v) / Math.log(base));
    return v > 0 && Math.abs(Math.pow(base, k) - v) <= 1e-6 * v ? k : null;
  }

  function powersIn(base, lo, hi) {
    var out = [];
    for (var k = Math.ceil(Math.log(lo) / Math.log(base) - 1e-9);
         Math.pow(base, k) <= hi * (1 + 1e-9); k++) {
      out.push(Math.pow(base, k));
    }
    return out;
  }

  // Axis numbers the way the deck writes them everywhere else -- 10k, 100k, 1M,
  // 1.5.  Deliberately the same rule as num() in scripts/chinchilla_svg.py, which
  // writes the SVG figures' axis numbers; the axes on both sides of a slide have
  // to agree about what a million looks like.
  function si(v) {
    var units = [[1e9, "B"], [1e6, "M"], [1e3, "k"]];
    for (var i = 0; i < units.length; i++) {
      if (Math.abs(v) >= units[i][0]) return trim(v / units[i][0]) + units[i][1];
    }
    return trim(v);
  }

  function trim(v) {
    return String(Math.round(v * 1e6) / 1e6);
  }

  // ------------------------------------------------------------- colours
  //
  // A series' colour has to be the *same* navy as the number quoting it in the
  // prose beside the plot, and the only way to say that once is the token block.
  // But colloquium reads `color:` in Python (elements/chart.py) and bakes it into
  // the canvas's JSON as borderColor, and Chart.js then hands the string straight
  // to a canvas context -- which resolves no custom properties and consults no
  // stylesheet, so a bare var() there would paint nothing at all.  This is where
  // it gets resolved instead, from the same :root the type and the stroke widths
  // come from.  Anything that is not a var() is passed through untouched, so a
  // hex, an rgb() or a named colour still works.
  //
  // Runs for every chart, with or without a `plot:` block: a fence that says
  // var(--deck-navy) must not depend on having asked for marker styling too.
  var COLOR_KEYS = ["borderColor", "backgroundColor", "pointBackgroundColor",
                    "pointBorderColor"];
  var VAR = /^var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)$/;

  function resolveColor(v) {
    var m = VAR.exec(String(v == null ? "" : v).trim());
    if (!m) return v;
    // Custom properties are substituted at computed-value time, so a token that
    // is itself a var() (--fig-surface -> --colloquium-bg) already reads as ink.
    var css = getComputedStyle(document.documentElement);
    return css.getPropertyValue(m[1]).trim() || (m[2] || "").trim() || v;
  }

  function resolveColors(chart) {
    (chart.config.data.datasets || []).forEach(function (ds) {
      COLOR_KEYS.forEach(function (key) {
        if (ds[key] != null) ds[key] = resolveColor(ds[key]);
      });
    });
  }

  // ------------------------------------------------------------ series pass
  //
  // tension 0 on every series, without exception: these are log-log
  // measurements read off a handful of points, and a spline would give them
  // curvature they did not earn.  colloquium's chart element hands out 0.3.
  //
  // A hollow marker is a *meaning* in this deck and nowhere else: on the alpha
  // sweep it marks a point that is still data-limited, so what is plotted is a
  // lower bound rather than a measurement.  Everything else is filled, which is
  // also why the finite-data and emergence charts cannot simply take
  // colloquium's default of a transparent fill.
  function styleSeries(chart, plot) {
    (chart.config.data.datasets || []).forEach(function (ds) {
      var label = ds.label || "";
      var guide = listed(label, plot.guide);
      var scatter = listed(label, plot.scatter);
      var marker = guide ? "none" : markerFor(plot, label);

      ds.tension = 0;
      ds.borderWidth = guide ? INK.hair : INK.data;
      if (guide || listed(label, plot.dash)) ds.borderDash = INK.dash;
      // Chart.js draws its datasets in reverse `order`, so the yardstick's 10 puts
      // it under the curves that are read against it and the scatter's 1 puts the
      // measured points on top of the predictions they are compared with.
      ds.order = guide ? 10 : (scatter ? 1 : 5);
      if (scatter) ds.showLine = false;

      if (marker === "none") {
        ds.pointRadius = 0;
      } else if (marker === "hollow") {
        ds.pointRadius = INK.point * HOLLOW_K;
        ds.pointBackgroundColor = INK.surface;
        ds.pointBorderColor = ds.borderColor;
        ds.pointBorderWidth = INK.data;
      } else {
        ds.pointRadius = INK.point * (scatter ? SCATTER_K : 1);
        ds.pointBackgroundColor = ds.borderColor;
        ds.pointBorderColor = ds.borderColor;
      }
    });
  }

  // The longest matching name wins, so a rule for "measured lower bound" can
  // override a broader one for "measured" without depending on key order.
  function markerFor(plot, label) {
    var over = plot.markerFor || {}, best = null;
    Object.keys(over).forEach(function (name) {
      if (hit(label, name) && (best === null || name.length > best.length)) best = name;
    });
    return best === null ? (plot.markers || "filled") : over[best];
  }

  // -------------------------------------------------------------- axis pass
  //
  // Powers of N are drawn on the canvas rather than written as text, and that is
  // worth keeping straight: the Unicode superscript digits are the obvious way to
  // write an exponent, but they are scattered over two code blocks and get picked
  // up from whichever fallback font happens to carry each one, so a plot's
  // exponents come out in two or three different faces.  Drawing them is the fix.
  // Chart.js still lays out an invisible label for each tick -- that is what
  // reserves the room the drawn one needs -- so the callback below returns the
  // plain-text form and the ink is set transparent.
  var DRAWN = {pow10: 10, pow2: 2};

  function ticksFor(kind, lo, hi) {
    if (kind === "si" || kind === "pow10") return powersIn(10, lo, hi);
    if (kind === "pow2") return powersIn(2, lo, hi);
    if (kind.indexOf("step:") === 0) {
      var step = parseFloat(kind.slice(5)), out = [];
      for (var i = 0; lo + i * step <= hi * (1 + 1e-9); i++) out.push(lo + i * step);
      return out;
    }
    if (kind.indexOf("at:") === 0) {
      return kind.slice(3).split(",").map(Number)
        .filter(function (v) { return v >= lo && v <= hi; });
    }
    return null;
  }

  function labelFor(kind, v) {
    if (kind === "si") return si(v);
    var base = DRAWN[kind];
    if (base) {
      var k = exponent(base, v);
      return k === null ? "" : String(base) + k;
    }
    return trim(v);
  }

  function wireAxis(chart, id, kind) {
    var scale = ((chart.config.options || {}).scales || {})[id];
    if (!scale || !kind) return;
    scale.afterBuildTicks = function (axis) {
      var want = ticksFor(kind, axis.min, axis.max);
      if (!want) return;
      axis.ticks = want.map(function (v) { return {value: v, major: true}; });
    };
    scale.ticks = Object.assign({autoSkip: false, maxRotation: 0}, scale.ticks, {
      color: DRAWN[kind] ? "transparent" : undefined,
      callback: function (v) { return labelFor(kind, v); },
    });
  }

  // The distance from an axis line to its labels, tick marks included.  Was
  // defined identically in three files.
  function tickGap(scale) {
    var grid = scale.options.grid || {}, ticks = scale.options.ticks || {};
    var len = grid.drawTicks === false ? 0 : (grid.tickLength == null ? 8 : grid.tickLength);
    return len + (ticks.padding == null ? 3 : ticks.padding);
  }

  // A mantissa plus a raised, smaller exponent, placed where Chart.js put the
  // invisible label it is standing in for.  The one copy of what used to be three.
  function drawPowers(chart, scale, base) {
    var ctx = chart.ctx, font = Chart.defaults.font, size = font.size;
    var mantissaFont = size + "px " + font.family;
    var exponentFont = Math.round(size * INK.exp) + "px " + font.family;
    ctx.save();
    ctx.fillStyle = Chart.defaults.color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    scale.ticks.forEach(function (tick, i) {
      var k = exponent(base, tick.value);
      if (k === null) return;
      var mantissa = String(base), power = String(k);
      ctx.font = mantissaFont;
      var wm = ctx.measureText(mantissa).width;
      ctx.font = exponentFont;
      var wp = ctx.measureText(power).width;
      var at = scale.getPixelForTick(i), x, y;
      if (scale.axis === "x") {
        x = at - (wm + wp) / 2;
        y = scale.top + tickGap(scale) + size / 2;
      } else {
        x = scale.right - tickGap(scale) - (wm + wp);
        y = at;
      }
      ctx.font = mantissaFont;
      ctx.fillText(mantissa, x, y);
      ctx.font = exponentFont;
      ctx.fillText(power, x + wm, y - size * 0.34);
    });
    ctx.restore();
  }

  // ------------------------------------------------------- in-plot notes
  // Variables in italic maths serif, everything else in the deck's body font --
  // the same split the SVG figures make with `.fig-var`, so an alpha looks the
  // same wherever in the deck it appears.

  function drawNotes(chart, notes) {
    var sx = chart.scales.x, sy = chart.scales.y;
    if (!sx || !sy) return;
    // An in-plot note is a figure's body text, not an axis label, so it takes
    // --fig-text where the axes take --fig-small.
    var ctx = chart.ctx, font = Chart.defaults.font;
    var size = Math.round(INK.noteSize);

    function fontFor(italic) {
      return italic
        ? "italic " + Math.round(size * INK.mathScale) + "px " + INK.mathFamily
        : size + "px " + font.family;
    }

    ctx.save();
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    notes.forEach(function (note) {
      var ds = (chart.data.datasets || []).filter(function (d) {
        return hit(d.label, note.series);
      })[0];
      if (!ds) return;
      // Odd segments of the _..._ split are the italic ones.
      var parts = String(note.text).split("_").map(function (t, i) {
        return {t: t, italic: i % 2 === 1};
      }).filter(function (p) { return p.t !== ""; });
      var width = 0;
      parts.forEach(function (p) {
        ctx.font = fontFor(p.italic);
        p.w = ctx.measureText(p.t).width;
        width += p.w;
      });
      var nudge = note.nudge || [0, 0];
      var x = sx.getPixelForValue(note.at[0]) + nudge[0] -
              (note.align === "right" ? width : 0);
      var y = sy.getPixelForValue(note.at[1]) + nudge[1];
      ctx.fillStyle = ds.borderColor;
      parts.forEach(function (p) {
        ctx.font = fontFor(p.italic);
        ctx.fillText(p.t, x, y);
        x += p.w;
      });
    });
    ctx.restore();
  }

  // ------------------------------------------------------------- canvas size
  //
  // Chart.js sizes a responsive canvas from its container's *rendered* rect, and
  // the deck is a fixed 1280x720 box that a CSS transform scales to the viewport,
  // so the canvas came out at the post-transform width: 488px at a 1280x720
  // window, 347px at the 800x600 one Chromium prints from.  On screen that is
  // nearly invisible, because the transform then scales the chart along with
  // everything else -- but print and PDF export do not use the canvas.  They use
  // the `<img>` the bootstrap snapshots it into, laid out at `width: 100%` of the
  // container's own 556px, so every exported chart was a 347px raster stretched
  // across 556px: type, lines and markers all ~1.6x too big and visibly soft next
  // to the page's hand-drawn SVG figures.  No font size can compensate for that;
  // it magnifies the whole canvas uniformly.
  //
  // The boxes these charts sit in are a fixed number of pixels and never change
  // size -- the deck's whole layout is fixed and only ever transformed -- so
  // responsiveness buys nothing here and is what introduced the dependence on the
  // viewport.  Taking the container's own layout box instead makes the canvas
  // agree with its container at every window size, on screen and in export.
  function pinSize(chart) {
    var box = chart.canvas.parentNode;
    // The bootstrap forces every slide visible while it builds the charts, so the
    // box has a size here even on slides that are not the current one.  If that
    // ever stops being true, fall through to Chart.js's own sizing.
    if (!box || !box.clientWidth || !box.clientHeight) return;
    chart.config.options.responsive = false;
    chart.width = box.clientWidth;
    chart.height = box.clientHeight;
  }

  // ------------------------------------------------------------------ plugin
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof Chart === "undefined") return;   // no chart on the page
    wireDefaults();
    Chart.register({
      id: "deckPlot",
      beforeInit: function (chart) {
        pinSize(chart);
        resolveColors(chart);
        var plot = spec(chart);
        if (!plot) return;
        styleSeries(chart, plot);
        wireAxis(chart, "x", plot.xTicks);
        wireAxis(chart, "y", plot.yTicks);
      },
      afterDraw: function (chart) {
        var plot = spec(chart);
        if (!plot) return;
        [["x", plot.xTicks], ["y", plot.yTicks]].forEach(function (pair) {
          var base = DRAWN[pair[1]];
          if (base && chart.scales[pair[0]]) drawPowers(chart, chart.scales[pair[0]], base);
        });
        if (plot.notes) drawNotes(chart, plot.notes);
      },
    });
  });

  // ------------------------------------------------------------- step sync
  //
  // One series per step.  A `.cap-cue` in slides.md names a series and sits in the
  // same `<!-- step -->` group as the paragraph that discusses it, so the line
  // arrives with its prose.  This is per-deck behaviour rather than anything the
  // `plot:` block can say -- the pairing lives in the slide, not in the figure --
  // so it is keyed off the presence of a cue and nothing else.  Series no cue names
  // (a yardstick, or the curve a revealed one is compared against) stay on
  // throughout, which is also what keeps the fixed axis bounds meaningful from the
  // first frame.
  var MAX_FRAMES = 120;   // ~2s: charts are built in a later "load" listener

  // A cue's name matches a series the same way every rule above does, by
  // substring, so `first 10k` governs both halves of a train/test pair at once.
  // When two cues match one series the longer name wins, so a pair can later be
  // split across two steps by adding the more specific cue, in either order.
  function cueFor(cues, label) {
    var best = null, bestName = null;
    cues.forEach(function (cue) {
      var name = cue.getAttribute("data-cap-series") || "";
      if (!hit(label, name)) return;
      if (bestName === null || name.length > bestName.length) {
        best = cue;
        bestName = name;
      }
    });
    return best;
  }

  function sync(chart, cues) {
    var changed = false;
    // Capture mode reveals every fragment from CSS alone, without the class the
    // runtime otherwise toggles, so the series have to follow suit.
    var forced = document.body.classList.contains("colloquium-capture");
    chart.data.datasets.forEach(function (ds, i) {
      var cue = cueFor(cues, ds.label || "");
      if (!cue) return;
      var holder = cue.closest("[data-fragment-index]");
      var shown = !holder || forced || holder.classList.contains("visible");
      if (chart.isDatasetVisible(i) === shown) return;
      chart.setDatasetVisibility(i, shown);
      changed = true;
    });
    if (changed) chart.update();
  }

  function wire(slide, frame) {
    var canvas = slide.querySelector("canvas[data-chart-config]");
    var cues = Array.prototype.slice.call(slide.querySelectorAll("[data-cap-series]"));
    if (!canvas || !cues.length) return true;   // nothing to do on this slide
    var chart = typeof Chart === "undefined" ? null : Chart.getChart(canvas);
    if (!chart) return false;                   // charts not constructed yet
    // A frame after building each chart, the bootstrap snapshots the canvas into
    // an <img> that print and PDF export use in its place.  Hiding series before
    // that would drop them from the export, so let the snapshot land first (and
    // give up waiting if it never does -- then the chart simply shows every
    // series, which is the old behaviour).
    if (frame < 10 && !canvas.parentNode.querySelector("img.colloquium-chart-print")) {
      return false;
    }
    sync(chart, cues);
    // presentation.js reveals a fragment by adding `visible` to it, so watching
    // the class covers every route in: arrows, slide jumps, the picker, capture.
    new MutationObserver(function () { sync(chart, cues); })
      .observe(slide, {subtree: true, attributes: true, attributeFilter: ["class"]});
    return true;
  }

  // One frame loop for both jobs, because they are ordered with respect to each
  // other: a slide's cues may not hide a series until that chart's snapshot is
  // current.
  function start(frame) {
    var pending = Array.prototype.slice.call(document.querySelectorAll(".slide"))
      .filter(function (s) {
        return s.querySelector("[data-cap-series]") && !s.hasAttribute("data-cap-wired");
      });
    pending.forEach(function (s) {
      if (wire(s, frame)) s.setAttribute("data-cap-wired", "1");
    });
    if (pending.length && frame < MAX_FRAMES) {
      requestAnimationFrame(function () { start(frame + 1); });
    }
  }

  window.addEventListener("load", function () { start(0); });
})();
