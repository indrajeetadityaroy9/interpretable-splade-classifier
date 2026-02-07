"""CLI entry point for the canonical interpretability benchmark via YAML."""

import argparse
import os
import yaml
from dataclasses import asdict
from transformers import AutoTokenizer

from splade.data.loader import load_sst2_data
from splade.evaluation.adversarial import CharacterAttack, TextFoolerAttack, WordNetAttack
from splade.evaluation.benchmark import (
    aggregate_results,
    benchmark_explainer,
    print_aggregated_results,
    print_interpretability_results,
)
from splade.evaluation.faithfulness import UnigramSampler
from splade.models.splade import SpladeModel
from splade.training.finetune import finetune_splade_for_ffidelity
from splade.utils.cuda import set_seed, DEVICE
from splade.config.load import load_config
from splade.config.schema import Config
from splade.training.loop import train_model
from splade.inference import score_model, explain_model
import torch

class PredictorWrapper:
    """Wrapper to make SpladeModel compatible with Predictor protocol."""
    def __init__(self, model, tokenizer, max_length, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        
    def predict_proba(self, texts):
        from splade.inference import predict_proba_model
        return predict_proba_model(self.model, self.tokenizer, texts, self.max_length, self.batch_size)

def run_benchmark(config: Config) -> list:
    """Run multi-seed benchmark defined by config."""
    all_seed_results = []
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)
    
    for seed in config.evaluation.seeds:
        print(f"\n" + "#" * 40)
        print(f"RUNNING BENCHMARK WITH SEED {seed}")
        print("#" * 40)
        
        set_seed(seed)

        train_texts, train_labels, test_texts, test_labels, num_labels = load_sst2_data(
            config.data.train_samples,
            config.data.test_samples,
            seed=seed,
        )
        config.data.num_labels = num_labels
        config.training.seed = seed
        
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        model = SpladeModel(config.model.name, config.data.num_labels).to(DEVICE)
        
        if config.model.compile:
            model = torch.compile(model, mode=config.model.compile_mode)
        
        train_model(model, tokenizer, train_texts, train_labels, config.training, config.model, config.data)
        
        accuracy = score_model(
            model, 
            tokenizer, 
            test_texts, 
            test_labels, 
            config.data.max_length, 
            config.training.batch_size, 
            config.data.num_labels
        )
        print(f"Model Accuracy: {accuracy:.4f}")
        mask_token = tokenizer.mask_token

        sampler = UnigramSampler(test_texts, seed=seed)
        print("\nFine-tuning model copy for F-Fidelity...")
        
        fine_tuned_model = finetune_splade_for_ffidelity(
            model,
            tokenizer,
            train_texts,
            train_labels,
            beta=config.evaluation.ffidelity_beta,
            ft_epochs=config.evaluation.ffidelity_ft_epochs,
            ft_lr=config.evaluation.ffidelity_ft_lr,
            batch_size=config.evaluation.ffidelity_ft_batch_size,
            mask_token=mask_token,
            seed=seed,
            max_length=config.data.max_length
        )
        
        predictor = PredictorWrapper(model, tokenizer, config.data.max_length, config.evaluation.batch_size)
        ft_predictor = PredictorWrapper(fine_tuned_model, tokenizer, config.data.max_length, config.evaluation.batch_size)

        attacks = [
            WordNetAttack(max_changes=config.evaluation.adversarial_max_changes),
            TextFoolerAttack(predictor, max_changes=config.evaluation.adversarial_max_changes),
            CharacterAttack(max_changes=config.evaluation.adversarial_max_changes),
        ]

        def splade_explain_fn(text, top_k):
            return explain_model(model, tokenizer, text, config.data.max_length, top_k=top_k, input_only=True)

        result = benchmark_explainer(
            predictor,
            "SPLADE",
            splade_explain_fn,
            test_texts,
            config.evaluation,
            mask_token,
            attacks=attacks,
            sampler=sampler,
            ftuned_clf=ft_predictor,
            tokenizer=tokenizer,
        )
        result.accuracy = accuracy
        results = [result]
        
        all_seed_results.append(results)

    if len(config.evaluation.seeds) > 1:
        aggregated = aggregate_results(all_seed_results)
        print_aggregated_results(aggregated)
        import json
        with open(os.path.join(config.output_dir, "metrics_aggregated.json"), "w") as f:
            json.dump(aggregated, f, indent=2)
    else:
        print_interpretability_results(all_seed_results[0], config.evaluation)
        
    return all_seed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical interpretability benchmark via YAML")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_benchmark(config)


if __name__ == "__main__":
    main()
