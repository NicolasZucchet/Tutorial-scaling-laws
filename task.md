# Developing the best associative memory model

## Problem

Use scaling laws to train the best associative memory toy model under compute constraints.

## Constraints

### Model

The model is $\hat{p}(\cdot \mid x) = \mathrm{softmax}(W x)$, with no bias. The output dimension is fixed to $d=512$.

### Data

- **Input statistics**: $p(x) \propto x^{-\gamma}$, with $\gamma = 1.2$, cut off at $p = 10^{-14}$. The embeddings are drawn uniformly at random from the unit sphere, and are resampled for each problem instance.
- **Output statistics**: $p(y \mid x)$ is a random distribution with entropy $H$ that is fixed over problem instances.
  For each $x$, the entropy $H$ is first sampled according to $\frac{\exp(H)}{d} \sim \mathrm{Beta}(1,31)$ so that the effective support size is $16$ on average.
  To sample a fixed distribution with desired entropy, sample logits iid from a unit Gaussian, and then find the softmax temperature that reaches the desired entropy.

### Evaluation

Don't use information about the structure of the problem to shortcut the answer.
  
### Optimizer

- Optimizer: Adam
* Learning-rate schedule: cosine, from max learning rate to a tenth of it 
* Batch size: 64
* Loss: cross-entropy

### Compute

Maximum $10^{13}$ flops for both tuning runs + final hero run. Use at most 3 screening rounds.

## Output

Produce a 1-page report with:

* Recipe
* Hero-run parameters
* Expected loss
* Actual loss

**You only have one shot.**
