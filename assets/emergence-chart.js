// Emergence chart (slide "Digression: emergent behavior under the hood"): marker
// styling, and power-of-ten tick labels on the model-size axis.
//
// Loaded from the <script src> tag at the end of figures/emergence-chart.md, which
// scripts/emergence.py writes together with the chart block; the tag carries no styling
// of its own, so this file is the single source of truth for how the chart looks.  It is
// identified by its y-axis title, which no other chart in the deck uses, so this file
// never touches the capacity, results or finite-data charts.
//
// The tag sits *after* the chart block because the marker pass below runs at parse time
// and only sees the canvases parsed so far.  Only one chart carries the tag today, but
// the tick plugin must not be registered twice if a second one ever does, so its
// registration is guarded by a flag on window.
(function () {
  var Y_TITLE = "top-1 accuracy";

  function isEmergenceChart(cfg) {
    var y = (((cfg.options || {}).scales || {}).y || {}).title || {};
    return y.text === Y_TITLE;
  }

  function xIsLog(cfg) {
    return ((((cfg.options || {}).scales || {}).x) || {}).type === "logarithmic";
  }

  // The model-size chart is nine measured models, so it gets filled markers on straight
  // segments, like the capacity and finite-data charts.  A dense over-training curve on a
  // linear x axis would get markerless lines instead, markers being only ink there.  No
  // tension either way: a sigmoid read off 9 points must not be given curvature it did
  // not earn.
  document.querySelectorAll("canvas[data-chart-config]").forEach(function (c) {
    var cfg = JSON.parse(c.getAttribute("data-chart-config"));
    if (!isEmergenceChart(cfg)) return;
    var r = xIsLog(cfg) ? 2.6 : 0;
    cfg.data.datasets.forEach(function (ds) {
      ds.tension = 0;
      ds.pointRadius = r;
      ds.pointBackgroundColor = ds.borderColor;
    });
    c.setAttribute("data-chart-config", JSON.stringify(cfg));
  });

  // Powers of ten on the model-size axis, drawn as a mantissa plus a raised, smaller
  // exponent -- same reason as the other two charts: the Unicode superscripts come from
  // whichever fallback font happens to carry them, so they do not match the text.
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

  if (window.__assocmemEmergenceTicks) return;
  window.__assocmemEmergenceTicks = true;

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof Chart === "undefined") return;
    Chart.register({
      id: "emergenceTicks",
      beforeInit: function (chart) {
        var cfg = chart.config;
        if (!isEmergenceChart(cfg) || !xIsLog(cfg)) return;
        var x = cfg.options.scales.x;
        x.afterBuildTicks = function (axis) {
          axis.ticks = axis.ticks.filter(function (t) {
            return exponent(t.value) !== null;
          });
        };
        x.ticks = Object.assign({autoSkip: false, maxRotation: 0}, x.ticks, {
          // the width of the invisible label reserves room for the drawn one
          color: "transparent",
          callback: function (v) { var k = exponent(v); return k === null ? "" : "10" + k; },
        });
      },
      afterDraw: function (chart) {
        if (!isEmergenceChart(chart.config) || !xIsLog(chart.config)) return;
        if (chart.scales.x) drawX(chart, chart.scales.x);
      },
    });
  });
})();
