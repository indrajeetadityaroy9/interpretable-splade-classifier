import yaml

from splade.config.schema import (Config, DataConfig, ModelConfig,
                                  TrainingConfig, VPEConfig)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        experiment_name=raw["experiment_name"],
        output_dir=raw["output_dir"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        seed=raw.get("seed", 42),
        training=TrainingConfig(**raw.get("training", {})),
        vpe=VPEConfig(**raw.get("vpe", {})),
    )
