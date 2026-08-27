<!-- Three columns, one per matrix multiplication a training step performs for a
     single linear layer: forward, backward-to-input, backward-to-weights.  Each
     is one matmul against W (or its transpose), so each costs 2N FLOPs for
     N = mn parameters -- one multiply and one add per entry of W.  Summing them
     gives the 6N FLOPs per parameter per token behind C = 6ND (Kaplan et al.,
     2020, appendix B). -->
<div class="training-flops" role="img"
     aria-label="Three matrix multiplications per training step -- forward, backward, and weight-gradient -- each costing 2N FLOPs, for 6N FLOPs per parameter per token and 6ND in total.">
  <div class="tf-card tf-forward">
    <div class="tf-step">1</div>
    <div class="tf-title">Forward</div>
    <div class="tf-purpose">make a prediction</div>
    <div class="tf-equation"><span class="math inline">y = Wx</span></div>
    <div class="tf-cost">2N FLOPs</div>
  </div>

  <div class="tf-card tf-backward fragment" data-colloquium-fragment="1">
    <div class="tf-step">2</div>
    <div class="tf-title">Backward</div>
    <div class="tf-purpose">propagate the error</div>
    <div class="tf-equation"><span class="math inline">\nabla_x L = W^\top \nabla_y L</span></div>
    <div class="tf-cost">2N FLOPs</div>
  </div>

  <div class="tf-card tf-learn fragment" data-colloquium-fragment="1">
    <div class="tf-step">3</div>
    <div class="tf-title">Learn</div>
    <div class="tf-purpose">accumulate weight gradients</div>
    <div class="tf-equation"><span class="math inline">\Delta W \mathrel{+}= \nabla_W L = \nabla_y L\,x^\top</span></div>
    <div class="tf-cost tf-cost-small">N products + N accumulations = 2N FLOPs</div>
  </div>

  <div class="tf-total fragment" data-colloquium-fragment="1">
    <span class="math inline">2N + 2N + 2N \to 6N</span> FLOPs per token, so <span class="math inline">6ND</span> FLOPs in total.
  </div>

</div>
