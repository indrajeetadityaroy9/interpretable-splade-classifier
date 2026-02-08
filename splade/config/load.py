import yaml

from splade.config.schema import (Config, DataConfig, EvaluationConfig,
                                  MechanisticConfig, ModelConfig)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    eval_raw = raw_config.get("evaluation", {})
    eval_kwargs = {}
    if "seeds" in eval_raw:
        eval_kwargs["seeds"] = eval_raw["seeds"]
    if "explainers" in eval_raw:
        eval_kwargs["explainers"] = eval_raw["explainers"]

    mech_raw = raw_config.get("mechanistic", {})
    # Drop legacy keys that no longer exist in MechanisticConfig
    mech_raw.pop("enabled", None)

    return Config(
        experiment_name=raw_config["experiment_name"],
        output_dir=raw_config["output_dir"],
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        evaluation=EvaluationConfig(**eval_kwargs),
        mechanistic=MechanisticConfig(**mech_raw),
    )
