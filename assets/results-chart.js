// Results charts: marker and dash styling. The alpha sweep also uses a hollow
// marker for a measurement that is still data-limited and is therefore only a bound.
//
// Referenced by the generated chart block in slides.md; scripts/grid_slide.py emits
// only the <script src> tag, so this file is the single source of truth for how that
// chart looks.  Kept separate from capacity-chart.js on purpose: that file owns the
// capacity slide and identifies its chart by a dataset label, so the two never meet.
(function () {
  // Two slides reference this file, so the browser runs it twice.  That is load-bearing,
  // not waste: the styling pass below runs at parse time and can only see the canvases
  // parsed so far, so the first slide's tag styles the first chart and the second slide's
  // tag styles both.  Re-styling is idempotent.  Registering the plugin is not, so only
  // that part is guarded.
  function isResultsChart(cfg) {
    return (cfg.data.datasets || []).some(function (ds) {
      return (ds.label || "").indexOf("theory") !== -1;
    });
  }

  // The measured points carry markers; the two curves are lines only, with the
  // theory dashed and behind, exactly as the capacity slide dashes its yardstick.
  document.querySelectorAll("canvas[data-chart-config]").forEach(function (c) {
    var cfg = JSON.parse(c.getAttribute("data-chart-config"));
    if (!isResultsChart(cfg)) return;
    cfg.data.datasets.forEach(function (ds) {
      var label = ds.label || "";
      ds.tension = 0;  // log-log data: straight segments, no invented curvature
      ds.pointBackgroundColor = ds.borderColor;
      if (label.indexOf("measured") !== -1) {
        ds.pointRadius = 3.4;
        ds.showLine = false;  // the measurement is points; the lines are predictions
        ds.order = 1;
        if (label.indexOf("lower bound") !== -1) {
          ds.pointBackgroundColor = "#fcfcfb";
          ds.pointBorderColor = ds.borderColor;
          ds.pointBorderWidth = 2;
          ds.pointRadius = 4.2;
        }
      } else if (label.indexOf("theory") !== -1) {
        ds.borderDash = [6, 5];
        ds.pointRadius = 0;
        ds.order = 10;
      } else {
        ds.pointRadius = 0;
        ds.order = 5;
      }
    });
    c.setAttribute("data-chart-config", JSON.stringify(cfg));
  });

  // Powers of ten on the compute axis, drawn as a mantissa plus a raised, smaller
  // exponent.  Same reason as on the capacity slide: the Unicode superscripts come
  // from whichever fallback font happens to carry them, so they do not match.
  function exponent(v) {
    var k = Math.round(Math.log(v) / Math.LN10);
    return v > 0 && Math.abs(Math.pow(10, k) - v) <= 1e-6 * v ? k : null;
  }

  function tickGap(scale) {
    var grid = scale.options.grid || {}, ticks = scale.options.ticks || {};
    var len = grid.drawTicks === false ? 0 : (grid.tickLength == null ? 8 : grid.tickLength);
    return len + (ticks.padding == null ? 3 : ticks.padding);
  }

  function drawX(chart, scale) {
    var ctx = chart.ctx, font = Chart.defaults.font, size = font.size;
    ctx.save();
    ctx.fillStyle = Chart.defaults.color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    scale.ticks.forEach(function (tick, i) {
      var k = exponent(tick.value);
      if (k === null) return;
      ctx.font = size + "px " + font.family;
      var wm = ctx.measureText("10").width;
      ctx.font = Math.round(size * 0.72) + "px " + font.family;
      var wp = ctx.measureText(String(k)).width;
      var at = scale.getPixelForTick(i);
      var x = at - (wm + wp) / 2, y = scale.top + tickGap(scale) + size / 2;
      ctx.font = size + "px " + font.family;
      ctx.fillText("10", x, y);
      ctx.font = Math.round(size * 0.72) + "px " + font.family;
      ctx.fillText(String(k), x + wm, y - size * 0.34);
    });
    ctx.restore();
  }

  // The alpha sweep names its two theory curves in the plot itself rather than in a
  // legend, the way the scaling-law figure labels its lines: each note is anchored to a
  // point on the curve it belongs to and nudged clear of it in pixels.  Chart.js has no
  // text mark, so they are drawn here, in the colour of the dataset they name.
  var CURVE_NOTES = [
    {series: "theory, model axis", x: 1.44, y: 0.44, dx: -8, dy: -18, align: "right",
     parts: [{t: "\u03b1", italic: true}, {t: " \u2212 1"}]},
    {series: "theory, data axis", x: 1.62, y: 0.3827, dx: 8, dy: 20, align: "left",
     parts: [{t: "1 \u2212 1/"}, {t: "\u03b1", italic: true}]},
  ];

  function hasSeries(chart, label) {
    return (chart.data.datasets || []).filter(function (ds) {
      return (ds.label || "") === label;
    })[0];
  }

  // Variables in italic math serif, everything else in the deck's body font -- the same
  // split the SVG figures make with `.pf-var`, so alpha looks the same wherever it appears.
  var MATH_FAMILY = '"Latin Modern Math", "Cambria Math", Georgia, serif';

  function drawCurveLabels(chart) {
    var sx = chart.scales.x, sy = chart.scales.y;
    if (!sx || !sy || sx.type !== "linear") return;   // compute-axis charts have no notes
    var ctx = chart.ctx, font = Chart.defaults.font;
    var size = Math.round(font.size * 1.15);

    function fontFor(part) {
      return part.italic ? "italic " + Math.round(size * 1.05) + "px " + MATH_FAMILY
                         : size + "px " + font.family;
    }

    ctx.save();
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    CURVE_NOTES.forEach(function (note) {
      var ds = hasSeries(chart, note.series);
      if (!ds) return;
      var width = 0;
      note.parts.forEach(function (part) {
        ctx.font = fontFor(part);
        part.w = ctx.measureText(part.t).width;
        width += part.w;
      });
      var x = sx.getPixelForValue(note.x) + note.dx - (note.align === "right" ? width : 0);
      var y = sy.getPixelForValue(note.y) + note.dy;
      ctx.fillStyle = ds.borderColor;
      note.parts.forEach(function (part) {
        ctx.font = fontFor(part);
        ctx.fillText(part.t, x, y);
        x += part.w;
      });
    });
    ctx.restore();
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof Chart === "undefined") return;
    if (window.__assocmemResultsTicks) return;
    window.__assocmemResultsTicks = true;
    Chart.register({
      id: "resultsTicks",
      beforeInit: function (chart) {
        var cfg = chart.config;
        if (!isResultsChart(cfg)) return;
        var scales = (cfg.options || {}).scales || {};
        // Only the compute axis is logarithmic and wants drawn 10^k labels; the alpha
        // sweep on the next slide reuses this file for its dash/marker styling but has
        // linear axes, where Chart.js's own labels are already right.
        if (scales.x && scales.x.type === "logarithmic") {
          scales.x.afterBuildTicks = function (axis) {
            axis.ticks = axis.ticks.filter(function (t) {
              return exponent(t.value) !== null;
            });
          };
          scales.x.ticks = Object.assign({autoSkip: false, maxRotation: 0},
                                         scales.x.ticks, {
            // the width of the invisible label reserves room for the drawn one
            color: "transparent",
            callback: function (v) { var k = exponent(v); return k === null ? "" : "10" + k; },
          });
        }
        // The loss axis spans well under a decade, so plain numbers are clearer than
        // powers of ten; only the tick *positions* need pinning.
        if (scales.y && scales.y.type === "logarithmic") {
          var want = [0.7, 0.8, 0.9, 1.0, 1.2, 1.5];
          scales.y.afterBuildTicks = function (axis) {
            axis.ticks = want
              .filter(function (v) { return v >= axis.min && v <= axis.max; })
              .map(function (v) { return {value: v, major: true}; });
          };
          scales.y.ticks = Object.assign({autoSkip: false}, scales.y.ticks, {
            callback: function (v) { return v.toFixed(1).replace(/\.0$/, ""); },
          });
        }
      },
      afterDraw: function (chart) {
        if (!isResultsChart(chart.config)) return;
        if (chart.scales.x && chart.scales.x.type === "logarithmic") {
          drawX(chart, chart.scales.x);
        }
        drawCurveLabels(chart);
      },
    });
  });
})();
