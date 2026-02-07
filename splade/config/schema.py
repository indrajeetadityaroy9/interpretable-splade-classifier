from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    dataset_name: str = "sst2"
    train_samples: int = 2000
    test_samples: int = 200
    num_labels: int = 2
    max_length: int = 128

@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    compile: bool = True
    compile_mode: str = "max-autotune"

@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 3
    base_lr: Optional[float] = None
    seed: int = 42
    df_alpha: float = 0.1
    df_beta: float = 5.0
    clip_factor: float = 0.01
    target_lambda_ratio: float = 0.5
    warmup_steps: Optional[int] = None
    num_workers: int = 4
    prefetch_factor: int = 2

@dataclass
class EvaluationConfig:
    batch_size: int = 32
    seeds: List[int] = field(default_factory=lambda: [42])
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    ffidelity_beta: float = 0.5
    ffidelity_ft_epochs: int = 3
    ffidelity_ft_lr: float = 1e-5
    ffidelity_ft_batch_size: int = 16
    monotonicity_steps: int = 10
    naopc_beam_size: int = 15
    soft_metric_n_samples: int = 20
    adversarial_mcp_threshold: float = 0.7
    adversarial_max_changes: int = 3
    adversarial_test_samples: int = 50
    ig_n_steps: int = 50
    lime_num_samples: int = 500

@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
