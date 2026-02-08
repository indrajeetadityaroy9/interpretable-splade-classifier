from dataclasses import dataclass, field


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
    circuit_threshold: float = 0.01
    sae_comparison: bool = False


@dataclass
class EvaluationConfig:
    seeds: list[int] = field(default_factory=lambda: [42])
    explainers: list[str] = field(default_factory=lambda: ["splade"])


@dataclass
class Config:
    experiment_name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    mechanistic: MechanisticConfig = field(default_factory=MechanisticConfig)
