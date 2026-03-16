import json
import os
import hydra
import torch
from omegaconf import DictConfig

from spalf.evaluation import evaluate_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    results = evaluate_checkpoint(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
