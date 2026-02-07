# Config Pruning Decision

## Kept (Migrated to YAML)

### Experiment
- `experiment_name`: Grouping tag.
- `output_dir`: Path for artifacts.

### Data
- `train_samples`: Crucial for data efficiency curves.
- `test_samples`: For consistent evaluation.
- `dataset_name`: Implicitly SST-2 now, made explicit for future proofing.
- `max_length`: Tokenization limit.

### Model
- `model_name`: Architecture backbone.
- `regularization`: Core contribution ("flops" vs "df_flops").
- `compile`: Toggle for H100 optimizations.

### Training
- `batch_size`: Efficiency/Performance knob.
- `max_epochs`: Training duration.
- `seed`: Reproducibility.
- `patience`: Early stopping control.
- `df_alpha`, `df_beta`: DF-FLOPS hyperparameters (Ablation targets).
- `target_lambda_ratio`: Sparsity control.
- `clip_factor`: Stability control.
- `base_lr`: Optimization hyperparameter.

### Evaluation (Benchmark Suite)
- `seeds`: Multi-seed evaluation.
- `k_values`: Metric parameter.
- `ffidelity_*`: All F-Fidelity parameters (critical for protocol).
- `monotonicity_steps`, `naopc_beam_size`: Metric precision.
- `soft_metric_n_samples`: Soft metric precision.
- `adversarial_*`: Robustness parameters.
- `ig_n_steps`, `lime_num_samples`: Baseline parameters.

## Removed / Fixed
- `num_workers`, `prefetch_factor`: Fixed to efficient defaults (4, 2) unless debugging, but can be exposed in `EfficiencyConfig`.
- `optimizer`: Fixed to AdamW (standard).
- `_EPS`: Numerical stability constants (fixed).
