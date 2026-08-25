// Slide "From capacity to scaling laws" (twin-axis version): the alpha slider
// redraws the two schematic log-log lines and moves their labels.
(function () {
  // Log-log window with real ticks: both lines start from the (1k, 1) corner,
  // one decade of N/D is DECADE_X across and one decade of L is DECADE_Y down.
  // So a unit of exponent is DECADES_X * DECADE_Y pixels of drop over the
  // plotted range, and the drawn slopes can be read off the ticks.  Steeper
  // lines simply leave the box (clipped), which is what a window does anyway.
  var X0 = 170, X1 = 890, Y0 = 100, Y_FLOOR = 400;
  var DECADE_X = 240, DECADE_Y = 120;
  var PX_PER_EXPONENT = ((X1 - X0) / DECADE_X) * DECADE_Y;

  function draw() {
    var slider = document.getElementById("sl-alpha");
    if (!slider) return;
    var a = parseFloat(slider.value);
    var sN = 1 - a;          // L(N) is proportional to N^(1-alpha)
    var sD = 1 / a - 1;      // L(D) is proportional to D^(1/alpha-1)
    var yN = Y0 - sN * PX_PER_EXPONENT;
    var yD = Y0 - sD * PX_PER_EXPONENT;

    document.getElementById("sl-line-n")
      .setAttribute("d", "M " + X0 + " " + Y0 + " L " + X1 + " " + yN.toFixed(1));
    document.getElementById("sl-line-d")
      .setAttribute("d", "M " + X0 + " " + Y0 + " L " + X1 + " " + yD.toFixed(1));

    // Each label follows its line to wherever the line leaves the box: off the
    // right edge for shallow slopes, through the floor for steep ones.
    var at = function (yEnd) {
      if (yEnd <= Y_FLOOR) return {x: X1 + 12, y: yEnd - 6};
      var xExit = X0 + (X1 - X0) * (Y_FLOOR - Y0) / (yEnd - Y0);
      return {x: xExit + 14, y: Y_FLOOR - 12};
    };
    var pN = at(yN), pD = at(yD);
    // Prise the pair apart when the two exponents are close enough to collide.
    if (Math.abs(pN.x - pD.x) < 90 && pN.y - pD.y < 30) {
      var mid = (pN.y + pD.y) / 2;
      pD.y = mid - 15;
      pN.y = mid + 15;
    }
    var place = function (id, pt) {
      var el = document.getElementById(id);
      el.setAttribute("x", pt.x.toFixed(1));
      el.setAttribute("y", pt.y.toFixed(1));
    };
    place("sl-label-n", pN);
    place("sl-label-d", pD);

    // The two exponents are readable off the ticks now, so the readout is just
    // the value being dragged.
    document.getElementById("sl-readout").innerHTML =
      '<span class="alpha-var">&#945;</span> = ' + a.toFixed(2);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var slider = document.getElementById("sl-alpha");
    if (!slider) return;
    slider.addEventListener("input", draw);
    // Arrow keys belong to the slider once it has focus, not to the deck.
    slider.addEventListener("keydown", function (e) { e.stopPropagation(); });
    draw();
  });
})();
