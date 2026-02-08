from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_name: str = "sst2"
    train_samples: int = 2000
    test_samples: int = 200


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"


@dataclass
class MechanisticConfig:
    enabled: bool = False
    circuit_threshold: float = 0.01
    ablation_samples: int = 100


@dataclass
class TrainingConfig:
    use_df_weighting: bool = True


@dataclass
class EvaluationConfig:
    seeds: List[int] = field(default_factory=lambda: [42])


@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    mechanistic: MechanisticConfig = field(default_factory=MechanisticConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
