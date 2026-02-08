import argparse
import json
import os
from dataclasses import asdict

import torch
import yaml
from transformers import AutoTokenizer

from splade.config.load import load_config
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.evaluation.mechanistic import (MechanisticResults,
                                           print_mechanistic_results,
                                           run_mechanistic_evaluation)
from splade.models.splade import SpladeModel
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mechanistic interpretability evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config.output_dir, exist_ok=True)

    for seed in config.evaluation.seeds:
        print(f"\n{'#' * 40}")
        print(f"MECHANISTIC EVAL WITH SEED {seed}")
        print("#" * 40)

        set_seed(seed)

        train_texts, train_labels, test_texts, test_labels, num_labels = load_dataset_by_name(
            config.data.dataset_name,
            config.data.train_samples,
            config.data.test_samples,
            seed=seed,
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        max_length = infer_max_length(train_texts, tokenizer)
        batch_size = _infer_batch_size(config.model.name, max_length)

        val_size = min(200, len(train_texts) // 5)
        val_texts_split = train_texts[-val_size:]
        val_labels_split = train_labels[-val_size:]
        train_texts_split = train_texts[:-val_size]
        train_labels_split = train_labels[:-val_size]

        model = SpladeModel(config.model.name, num_labels).to(DEVICE)
        model = torch.compile(model, mode="reduce-overhead")

        train_model(
            model, tokenizer, train_texts_split, train_labels_split,
            model_name=config.model.name, num_labels=num_labels,
            val_texts=val_texts_split, val_labels=val_labels_split,
            use_df_weighting=config.training.use_df_weighting,
        )

        # Prepare test inputs as tensors
        encoding = tokenizer(
            test_texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids_list = [
            encoding["input_ids"][i:i+1].to(DEVICE)
            for i in range(len(test_texts))
        ]
        attention_mask_list = [
            encoding["attention_mask"][i:i+1].to(DEVICE)
            for i in range(len(test_texts))
        ]

        results = run_mechanistic_evaluation(
            model,
            input_ids_list,
            attention_mask_list,
            test_labels,
            tokenizer,
            num_classes=num_labels,
            circuit_threshold=config.mechanistic.circuit_threshold,
        )

        print_mechanistic_results(results)


if __name__ == "__main__":
    main()
