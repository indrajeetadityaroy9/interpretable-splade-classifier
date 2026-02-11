from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset_name: str = "banking77"
    train_samples: int = -1  # -1 = use full split
    test_samples: int = -1   # -1 = use full split


@dataclass
class ModelConfig:
    name: str = "answerdotai/ModernBERT-base"


@dataclass
class TrainingConfig:
    sparsity_target: float = 0.1
    warmup_fraction: float = 0.2
    pooling: str = "max"  # "max" or "attention"
    learning_rate: float = 3e-4  # Schedule-Free AdamW base LR


@dataclass
class VPEConfig:
    enabled: bool = False
    token_ids: list[int] = field(default_factory=list)
    num_senses: int = 4


@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vpe: VPEConfig = field(default_factory=VPEConfig)
