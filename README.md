# SPALF: Constrained Sparse Autoencoders via Augmented Lagrangian with Adaptive Dual Updates

Standard sparse autoencoders (SAEs) minimize a weighted sum of reconstruction error and sparsity, requiring manual tuning of the sparsity coefficient. SPALF removes this tuning by formulating SAE training as constrained optimization: minimize $\ell_0$ sparsity subject to reconstruction, decoder alignment, and feature orthogonality constraints, solved via an augmented Lagrangian with adaptive penalty weights and dual variable updates.

## Method

### Two-Block Decoder

The decoder splits into two blocks:

- A **vocabulary-aligned** block $W_A \in \mathbb{R}^{d \times V}$, initialized from the model's unembedding matrix and kept close to it through a decoder alignment constraint. Each feature in this block corresponds to a token in the vocabulary.
- A **learned** block $W_B \in \mathbb{R}^{d \times (F-V)}$, trained from scratch with unit-norm columns to capture structure not explained by vocabulary directions.

The encoder rows for the vocabulary-aligned block are initialized as whitened vocabulary vectors $W_{\text{enc}}^{[:V]} = (\Sigma^{-1/2} W_{\text{vocab}})^\top$, so each pre-activation computes a whitened dot product with the corresponding vocabulary direction. The remaining encoder rows are orthogonalized against the vocabulary subspace via QR decomposition. The decoder bias is set to the activation mean.

### Smooth JumpReLU Activation

Features are activated by JumpReLU with learnable per-feature thresholds. The step function in JumpReLU is non-differentiable, so during backpropagation it is replaced by a smooth surrogate based on the Moreau envelope: within a transition zone $(-\sqrt{2\gamma_j},\, 0]$ around each threshold, the gradient is a linear ramp $-u/\gamma_j$, giving a continuously differentiable approximation to the step function. The per-feature bandwidth $\gamma_j$ is calibrated from the interquartile range of pre-activations near each threshold.

Unlike the piecewise-constant straight-through estimator in the JumpReLU SAE ([arXiv:2407.14435](https://arxiv.org/abs/2407.14435)), this surrogate is smooth at the decision boundary. Both the $\ell_0$ objective and the constraints receive gradients through this surrogate via fused Triton kernels.

### Augmented Lagrangian Optimization

The training objective is an augmented Lagrangian with a smooth penalty function ([arXiv:2510.20995](https://arxiv.org/abs/2510.20995)):

$$\Psi(g, y) = \frac{(\max(0,\, 2g + y))^2 - y^2}{4}$$

which provides gradients even when a constraint is inactive, unlike the standard quadratic penalty. Dual variables (Lagrange multipliers) are updated with a proportional-integral rule $\lambda \leftarrow [\lambda + \rho(2v - v_{\text{prev}})]_+$, equivalent to gradient descent-ascent on the augmented Lagrangian ([arXiv:2509.22500](https://arxiv.org/abs/2509.22500)).

Each constraint has its own penalty weight $\rho_i$ that adapts via non-monotone updates ([arXiv:2412.14269](https://arxiv.org/abs/2412.14269)): $\rho_i$ increases when violations are large and decreases when constraints are satisfied, with a per-constraint floor guaranteeing convergence. Constraint violations are smoothed with a triple exponential moving average ([arXiv:2306.01423](https://arxiv.org/abs/2306.01423)):

$$\hat{v} = 3 \cdot \text{EMA}_1 - 3 \cdot \text{EMA}_2 + \text{EMA}_3$$

which reduces tracking lag by a factor of $(1-\beta)^2$ compared to a single EMA, with no extra hyperparameters.

### Surrogate Bandwidth Annealing

The surrogate bandwidth $\gamma$ controls how closely the smooth activation approximates the true step function. It is annealed using the transition-zone mass:

$$D(t) = \mathbb{E}\!\left[\frac{u^2}{\gamma} \cdot \mathbf{1}_{u \in (-\sqrt{2\gamma},\, 0]}\right]$$

which measures the fraction of features currently in the smooth transition zone. As training progresses and features separate from their thresholds, $D \to 0$ and $\gamma \to \gamma_{\text{floor}}$, so the surrogate converges to the true step function.

The smoothed ratio $D_{\text{ema}} / D_0$ drives two schedules:

1. **Bandwidth.** $\gamma = \max(D_{\text{ema}}/D_0,\; \alpha_{\text{floor}}) \cdot \gamma_{\text{init}}$, with a time-dependent floor $\gamma_{\text{floor}}(t) = O(t^{-1/3})$ that preserves the variable-smoothing convergence rate ([arXiv:2003.07612](https://arxiv.org/abs/2003.07612)).
2. **Discretization correction.** A coefficient $\lambda_{\text{disc}}$ grows monotonically to close the gap between the surrogate $\ell_0$ and the true $\ell_0$.

Once all smoothed constraint violations become feasible, a KL-divergence constraint activates, replacing the reconstruction-based fidelity target with the distributional match between original and SAE-patched next-token predictions.

### Online Whitening

Activations are whitened online using a Frequent Directions streaming covariance sketch ([arXiv:1501.01711](https://arxiv.org/abs/1501.01711)) that maintains an $O(\sqrt{d} \times d)$ low-rank approximation via incremental SVD. The sketch is regularized with Ledoit-Wolf nonlinear shrinkage for numerically stable inversion. The resulting whitening transform provides the norm used in both the reconstruction constraint and the encoder initialization.
