"""CIS experiment pipeline: Train -> Evaluation -> Integration.

Three-phase pipeline producing a unified ExperimentResults artifact.
All CIS mechanisms (DF-FLOPS, circuit losses, DLA, W_eff) are always active.
"""

import argparse
import json
import os
from dataclasses import asdict

import torch
import yaml

from splade.config.load import load_config
from splade.config.schema import Config
from splade.evaluation.benchmark import (ExperimentResults,
                                         InterpretabilityResult,
                                         aggregate_results,
                                         benchmark_explainer,
                                         print_aggregated_results,
                                         print_experiment_results)
from splade.evaluation.constants import K_MAX
from splade.evaluation.explainers import EXPLAINER_REGISTRY
from splade.evaluation.f_fidelity import finetune_surrogate_model
from splade.evaluation.faithfulness import UnigramSampler
from splade.evaluation.integration import analyze_circuit_faithfulness_alignment
from splade.evaluation.mechanistic import (print_mechanistic_results,
                                           run_mechanistic_evaluation)
from splade.evaluation.token_alignment import normalize_attributions_to_words
from splade.inference import explain_model, explain_model_batch
from splade.pipelines import (PredictorWrapper, prepare_mechanistic_inputs,
                              setup_and_train)
from splade.training.constants import FRAMEWORK_NAME
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE


def _print_cis_config(config: Config) -> None:
    """Print CIS experiment configuration."""
    print(f"\n--- {FRAMEWORK_NAME} ---")
    print(f"  Model:       {config.model.name}")
    print(f"  Dataset:     {config.data.dataset_name} (train={config.data.train_samples}, test={config.data.test_samples})")
    print(f"  Seeds:       {config.evaluation.seeds}")
    print(f"  Explainers:  {config.evaluation.explainers}")
    print(f"  SAE compare: {'yes' if config.mechanistic.sae_comparison else 'no'}")
    print()


def run_experiment(config: Config) -> list[ExperimentResults]:
    """Run the full CIS experiment pipeline across all seeds."""
    all_results = []

    _print_cis_config(config)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    for seed in config.evaluation.seeds:
        print(f"\n{'#' * 60}")
        print(f"EXPERIMENT SEED {seed}")
        print(f"{'#' * 60}")

        # Phase 1: CIS Training
        print("\n--- PHASE 1: CIS TRAINING ---")
        exp = setup_and_train(config, seed)
        print(f"Accuracy: {exp.accuracy:.4f}")

        eval_batch_size = min(exp.batch_size * 4, 128)
        experiment = ExperimentResults(
            config_snapshot=asdict(config),
            seed=seed,
            accuracy=exp.accuracy,
        )

        # Phase 2: Evaluation (Mechanistic + Faithfulness)
        print("\n--- PHASE 2: EVALUATION ---")

        input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
            exp.tokenizer, exp.test_texts, exp.max_length,
        )

        mechanistic_results = run_mechanistic_evaluation(
            exp.model, input_ids_list, attention_mask_list,
            exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
            circuit_threshold=config.mechanistic.circuit_threshold,
            run_sae_comparison=config.mechanistic.sae_comparison,
        )

        experiment.dla_verification_error = mechanistic_results.dla_verification_error
        experiment.circuits = mechanistic_results.circuits
        experiment.circuit_completeness = mechanistic_results.circuit_completeness
        experiment.semantic_fidelity = mechanistic_results.semantic_fidelity
        experiment.sae_comparison = mechanistic_results.sae_comparison

        print_mechanistic_results(mechanistic_results)

        mask_token = exp.tokenizer.mask_token
        sampler = UnigramSampler(exp.test_texts, seed=seed)
        predictor = PredictorWrapper(
            exp.model, exp.tokenizer, exp.max_length, eval_batch_size,
        )

        surrogate_model = finetune_surrogate_model(
            exp.model, exp.tokenizer, exp.train_texts, exp.train_labels,
            exp.max_length, seed=seed,
        )

        attributions_per_explainer: dict[str, list[list[tuple[str, float]]]] = {}

        for explainer_name in config.evaluation.explainers:
            if explainer_name == "splade":
                def splade_explain_fn(text, top_k):
                    return explain_model(
                        exp.model, exp.tokenizer, text, exp.max_length,
                        top_k=top_k, input_only=True,
                    )

                def splade_batch_explain_fn(texts, top_k):
                    return explain_model_batch(
                        exp.model, exp.tokenizer, texts, exp.max_length,
                        top_k=top_k, input_only=True, batch_size=eval_batch_size,
                    )

                explain_fn = splade_explain_fn
                batch_explain_fn = splade_batch_explain_fn
                display_name = "SPLADE"
            else:
                explainer_obj = EXPLAINER_REGISTRY[explainer_name](seed=seed)

                def _make_explain(obj=explainer_obj):
                    def fn(text, top_k):
                        return obj.explain(exp.model, exp.tokenizer, text, exp.max_length, top_k)
                    return fn

                def _make_batch_explain(obj=explainer_obj):
                    def fn(texts, top_k):
                        return obj.explain_batch(exp.model, exp.tokenizer, texts, exp.max_length, top_k, batch_size=eval_batch_size)
                    return fn

                explain_fn = _make_explain()
                batch_explain_fn = _make_batch_explain()
                display_name = explainer_name.upper()

            result = benchmark_explainer(
                predictor, display_name, explain_fn, batch_explain_fn,
                exp.test_texts, mask_token=mask_token, seed=seed,
                sampler=sampler, tokenizer=exp.tokenizer, max_length=exp.max_length,
                surrogate_model=surrogate_model, test_labels=exp.test_labels,
            )
            result.accuracy = exp.accuracy
            experiment.explainer_results.append(result)

            raw = batch_explain_fn(exp.test_texts, K_MAX)
            attributions_per_explainer[display_name] = [
                normalize_attributions_to_words(text, attrib, exp.tokenizer)
                for text, attrib in zip(exp.test_texts, raw)
            ]

        # Phase 3: Integration Analysis
        if mechanistic_results.circuits:
            print("\n--- PHASE 3: CIRCUIT-FAITHFULNESS INTEGRATION ---")

            experiment.circuit_faithfulness_alignment = analyze_circuit_faithfulness_alignment(
                mechanistic_results.circuits,
                experiment.explainer_results,
                attributions_per_explainer,
                exp.test_labels,
            )

        print_experiment_results(experiment)
        all_results.append(experiment)

    _save_results(config, all_results)

    return all_results


def _save_results(config: Config, results: list[ExperimentResults]) -> None:
    """Save experiment results to JSON."""
    serializable = []
    for r in results:
        d = {
            "seed": r.seed,
            "accuracy": r.accuracy,
            "dla_verification_error": r.dla_verification_error,
            "semantic_fidelity": r.semantic_fidelity,
            "sae_comparison": r.sae_comparison,
            "circuit_faithfulness_alignment": r.circuit_faithfulness_alignment,
        }
        # Circuits: serialize VocabularyCircuit objects
        circuits_ser = {}
        for class_idx, circuit in r.circuits.items():
            circuits_ser[str(class_idx)] = {
                "token_ids": circuit.token_ids,
                "token_names": circuit.token_names,
                "attribution_scores": circuit.attribution_scores,
                "completeness_score": circuit.completeness_score,
            }
        d["circuits"] = circuits_ser
        d["circuit_completeness"] = {str(k): v for k, v in r.circuit_completeness.items()}

        # Explainer results
        d["explainer_results"] = []
        for er in r.explainer_results:
            er_dict = {
                "name": er.name,
                "accuracy": er.accuracy,
                "soft_comprehensiveness": er.soft_comprehensiveness,
                "soft_sufficiency": er.soft_sufficiency,
                "causal_faithfulness": er.causal_faithfulness,
                "monotonicity": er.monotonicity,
                "f_fidelity_pos": er.f_fidelity_pos,
                "f_fidelity_neg": er.f_fidelity_neg,
                "adversarial_sensitivity": er.adversarial_sensitivity,
                "inference_latency": er.inference_latency,
                "filler_comprehensiveness": {str(k): v for k, v in er.filler_comprehensiveness.items()},
                "eraser_comprehensiveness": {str(k): v for k, v in er.eraser_comprehensiveness.items()},
                "eraser_sufficiency": {str(k): v for k, v in er.eraser_sufficiency.items()},
                "aopc": {str(k): v for k, v in er.aopc.items()},
                "naopc": {str(k): v for k, v in er.naopc.items()},
            }
            d["explainer_results"].append(er_dict)

        serializable.append(d)

    output_path = os.path.join(config.output_dir, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Multi-seed aggregation
    if len(results) > 1:
        all_explainer_results = [r.explainer_results for r in results]
        aggregated = aggregate_results(all_explainer_results)
        print_aggregated_results(aggregated)
        with open(os.path.join(config.output_dir, "metrics_aggregated.json"), "w") as f:
            json.dump(aggregated, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CIS experiment: Train -> Evaluation -> Integration",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
