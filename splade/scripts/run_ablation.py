"""Ablation study: DF-FLOPS vs Vanilla FLOPS impact on mechanistic interpretability."""

import argparse
import os

import torch
from transformers import AutoTokenizer

from splade.config.load import load_config
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.evaluation.mechanistic import (print_mechanistic_results,
                                           run_mechanistic_evaluation)
from splade.models.splade import SpladeModel
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DF-FLOPS vs Vanilla FLOPS ablation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.evaluation.seeds[0]
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

    encoding = tokenizer(
        test_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids_list = [encoding["input_ids"][i:i+1].to(DEVICE) for i in range(len(test_texts))]
    attention_mask_list = [encoding["attention_mask"][i:i+1].to(DEVICE) for i in range(len(test_texts))]

    for variant_name, use_df in [("DF-FLOPS", True), ("Vanilla-FLOPS", False)]:
        print(f"\n{'='*60}")
        print(f"Training with {variant_name}")
        print(f"{'='*60}")

        set_seed(seed)
        model = SpladeModel(config.model.name, num_labels).to(DEVICE)
        model = torch.compile(model, mode="reduce-overhead")

        train_model(
            model, tokenizer, train_texts_split, train_labels_split,
            model_name=config.model.name, num_labels=num_labels,
            val_texts=val_texts_split, val_labels=val_labels_split,
            use_df_weighting=use_df,
        )

        results = run_mechanistic_evaluation(
            model, input_ids_list, attention_mask_list,
            test_labels, tokenizer, num_classes=num_labels,
        )
        print(f"\n--- {variant_name} Results ---")
        print_mechanistic_results(results)


if __name__ == "__main__":
    main()
