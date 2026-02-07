# Configuration Inventory

## CLI Arguments
### `splade/scripts/train.py`
- `--train-samples`: int (2000)
- `--test-samples`: int (200)
- `--epochs`: int (None)
- `--max-epochs`: int (20)
- `--batch-size`: int (64)
- `--seed`: int (42)
- `--output-dir`: str ("checkpoints")
- `--regularization`: str ("flops")

### `splade/scripts/eval.py`
- `--train-samples`: int (2000)
- `--test-samples`: int (200)
- `--epochs`: int (2)
- `--batch-size`: int (32)
- `--seeds`: list[int] ([42])
- `--regularization`: str ("flops")

## Config Classes
### `splade/evaluation/benchmark.py:BenchmarkConfig`
- `seed`: 42
- `k_values`: (1, 5, 10, 20)
- `ffidelity_beta`: 0.5
- `ffidelity_ft_epochs`: 3
- `ffidelity_ft_lr`: 1e-5
- `ffidelity_ft_batch_size`: 16
- `monotonicity_steps`: 10
- `naopc_beam_size`: 15
- `soft_metric_n_samples`: 20
- `adversarial_mcp_threshold`: 0.7
- `adversarial_max_changes`: 3
- `adversarial_test_samples`: 50
- `ig_n_steps`: 50
- `lime_num_samples`: 500

## Hardcoded Constants & Hidden Knobs
- `splade/models/classifier.py`:
  - `model_name`: "distilbert-base-uncased" (in __init__ default)
  - `max_length`: 128 (in __init__ default)
  - `df_alpha`: 0.1 (in __init__ default)
  - `df_beta`: 5.0 (in __init__ default)
  - `torch.compile(mode="max-autotune")`
  - `DataLoader(num_workers=4, prefetch_factor=2)`
  - `AdamW` optimizer
- `splade/training/optim.py`:
  - `_compute_base_lr`: `2e-5 * (768/hidden)`
  - `_adaptive_gradient_clip`: `clip_factor=0.01`, `eps=1e-3`
  - `_AdaptiveLambdaSchedule`: `target_ratio=0.5`, `ema_decay=0.99`, `lambda_ema_decay=0.95`, `max_lambda_change=2.0`
  - `_EarlyStopping`: `patience=3`, `min_delta=1e-4`
