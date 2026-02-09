"""Multi-dataset CIS benchmark: SST-2, AG News, e-SNLI.

Runs the full CIS pipeline on each dataset and produces a combined
results table for paper evaluation.
"""

import argparse
import copy
import json
import os

from splade.config.load import load_config
from splade.scripts.run_experiment import run_experiment


PAPER_DATASETS = {
    "sst2": {"train_samples": -1, "test_samples": -1},
    "ag_news": {"train_samples": -1, "test_samples": -1},
    "imdb": {"train_samples": -1, "test_samples": -1},
}


def run_multi_dataset(config) -> dict[str, list[dict]]:
    """Run full CIS pipeline on each dataset, return combined results."""
    all_results = {}
    base_output = config.output_dir

    for dataset_name, sizes in PAPER_DATASETS.items():
        print(f"\n{'#' * 60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'#' * 60}")

        ds_config = copy.deepcopy(config)
        ds_config.data.dataset_name = dataset_name
        ds_config.data.train_samples = sizes["train_samples"]
        ds_config.data.test_samples = sizes["test_samples"]
        ds_config.output_dir = os.path.join(base_output, dataset_name)
        ds_config.experiment_name = f"cis_{dataset_name}"

        results = run_experiment(ds_config)
        all_results[dataset_name] = results

    _print_combined_table(all_results)
    _save_combined(base_output, all_results)

    return all_results


def _print_combined_table(all_results: dict[str, list[dict]]) -> None:
    print(f"\n{'=' * 100}")
    print("MULTI-DATASET RESULTS")
    print(f"{'=' * 100}")
    header = (
        f"{'Dataset':<12} {'Acc':>8} {'DLA Err':>10} "
        f"{'AOPC-C':>8} {'AOPC-S':>8} "
        f"{'Jac Sep':>10}"
    )
    print(header)
    print("-" * 100)

    for dataset_name, results_list in all_results.items():
        for r in results_list:
            eraser = r.get("eraser_metrics", {})
            sf = r.get("semantic_fidelity", {})
            print(
                f"{dataset_name:<12} "
                f"{r['accuracy']:>8.4f} "
                f"{r['dla_verification_error']:>10.6f} "
                f"{eraser.get('aopc_comprehensiveness', 0):>8.4f} "
                f"{eraser.get('aopc_sufficiency', 0):>8.4f} "
                f"{sf.get('class_separation', 0):>10.4f}"
            )
    print(f"{'=' * 100}")


def _save_combined(output_dir: str, all_results: dict[str, list[dict]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "multi_dataset_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIS on multiple datasets")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_multi_dataset(config)


if __name__ == "__main__":
    main()
