// Capacity chart (slide "Capacity, in theory and in practice"): marker/dash
// styling and hand-drawn power-of-two / power-of-ten tick labels.
//
// Referenced by the generated chart block in slides.md; scripts/capacity_sweep.py
// emits only the <script src> tag, so this file is the single source of truth.
(function () {
  function exponent(base, v) {   // k with base^k == v, or null
    var k = Math.round(Math.log(v) / Math.log(base));
    return v > 0 && Math.abs(Math.pow(base, k) - v) <= 1e-6 * v ? k : null;
  }
  function isCapacityChart(cfg) {
    return (cfg.data.datasets || []).some(function (ds) {
      return (ds.label || "").indexOf("parameter") !== -1;
    });
  }

  // Small filled markers, and the slope-1 yardstick dashed and in the back.
  document.querySelectorAll("canvas[data-chart-config]").forEach(function (c) {
    var cfg = JSON.parse(c.getAttribute("data-chart-config"));
    if (!isCapacityChart(cfg)) return;
    cfg.data.datasets.forEach(function (ds) {
      ds.tension = 0;          // log-log data: straight segments, no invented curvature
      ds.pointRadius = 2.5;
      ds.pointBackgroundColor = ds.borderColor;   // filled markers, not rings
      if ((ds.label || "").indexOf("parameter") !== -1) {
        ds.borderDash = [6, 5];  // slope-1 yardstick: dashed, no markers, in the back
        ds.pointRadius = 0;
        ds.order = 10;
      }
    });
    c.setAttribute("data-chart-config", JSON.stringify(cfg));
  });

  // Ticks only where there is a measurement -- powers of two on x, powers of ten on
  // y -- and their labels drawn as a mantissa plus a raised, smaller exponent.  The
  // Unicode superscript digits are the obvious way to write those, but they are
  // scattered over two code blocks and get picked up from whichever fallback font
  // has them, so the exponents come out mismatched; drawing them is the fix.  Chart.js
  // still lays the labels out (that is what reserves the space for them), just in
  // transparent ink.
  var BASE = {x: 2, y: 10};

  function tickGap(scale) {   // axis line to label, tick marks included
    var grid = scale.options.grid || {}, ticks = scale.options.ticks || {};
    var len = grid.drawTicks === false ? 0 : (grid.tickLength == null ? 8 : grid.tickLength);
    return len + (ticks.padding == null ? 3 : ticks.padding);
  }

  function drawTickLabels(chart, scale) {
    var base = BASE[scale.axis];
    var ctx = chart.ctx;
    var font = Chart.defaults.font;
    var size = font.size;
    var mantissaFont = size + "px " + font.family;
    var exponentFont = Math.round(size * 0.72) + "px " + font.family;
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

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof Chart === "undefined") return;
    Chart.register({
      id: "capacityTicks",
      beforeInit: function (chart) {
        var cfg = chart.config;
        if (!isCapacityChart(cfg)) return;
        var scales = (cfg.options || {}).scales || {};
        var measured = [];
        cfg.data.datasets.forEach(function (ds) {
          (ds.data || []).forEach(function (pt) {
            if (measured.indexOf(pt.x) === -1) measured.push(pt.x);
          });
        });
        measured.sort(function (u, v) { return u - v; });
        if (scales.x) {
          scales.x.afterBuildTicks = function (axis) {
            axis.ticks = measured
              .filter(function (v) { return v >= axis.min && v <= axis.max; })
              .map(function (v) { return {value: v, major: true}; });
          };
          scales.x.ticks = Object.assign({autoSkip: false, maxRotation: 0},
                                         scales.x.ticks, {
            // the width of the invisible label is what reserves room for the drawn one
            color: "transparent",
            callback: function (v) { var k = exponent(2, v); return k === null ? v : "2" + k; },
          });
        }
        if (scales.y) {
          scales.y.afterBuildTicks = function (axis) {
            axis.ticks = axis.ticks.filter(function (t) { return exponent(10, t.value) !== null; });
          };
          scales.y.ticks = Object.assign({autoSkip: false}, scales.y.ticks, {
            color: "transparent",
            callback: function (v) { var k = exponent(10, v); return k === null ? "" : "10" + k; },
          });
        }
      },
      afterDraw: function (chart) {
        if (!isCapacityChart(chart.config)) return;
        Object.keys(BASE).forEach(function (id) {
          if (chart.scales[id]) drawTickLabels(chart, chart.scales[id]);
        });
      },
    });
  });
})();


// One series per step: a `.cap-cue` in the markdown names a dataset and sits in
// the same `<!-- step -->` group as the paragraph that discusses it, so the line
// arrives with its prose.  Datasets no cue names (the slope-1 yardstick) stay on
// throughout, which also keeps the fixed axis bounds meaningful from the start.
(function () {
  var MAX_FRAMES = 120;   // ~2s: charts are built in a later "load" listener

  function sync(chart, cues) {
    var changed = false;
    // Capture mode reveals every fragment from CSS alone, without the class the
    // runtime otherwise toggles, so the series have to follow suit.
    var forced = document.body.classList.contains("colloquium-capture");
    cues.forEach(function (cue) {
      var holder = cue.closest("[data-fragment-index]");
      var shown = !holder || forced || holder.classList.contains("visible");
      var wanted = (cue.getAttribute("data-cap-series") || "").toLowerCase();
      chart.data.datasets.forEach(function (ds, i) {
        if ((ds.label || "").toLowerCase().indexOf(wanted) === -1) return;
        if (chart.isDatasetVisible(i) === shown) return;
        chart.setDatasetVisibility(i, shown);
        changed = true;
      });
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
