import yaml

from splade.config.schema import (Config, DataConfig, EvaluationConfig,
                                  MechanisticConfig, ModelConfig,
                                  TrainingConfig)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    eval_raw = raw_config.get("evaluation", {})
    eval_kwargs = {}
    if "seeds" in eval_raw:
        eval_kwargs["seeds"] = eval_raw["seeds"]

    mech_raw = raw_config.get("mechanistic", {})
    train_raw = raw_config.get("training", {})

    return Config(
        experiment_name=raw_config["experiment_name"],
        output_dir=raw_config["output_dir"],
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        evaluation=EvaluationConfig(**eval_kwargs),
        mechanistic=MechanisticConfig(**mech_raw),
        training=TrainingConfig(**train_raw),
    )
