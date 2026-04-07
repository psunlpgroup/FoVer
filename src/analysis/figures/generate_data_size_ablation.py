import json
from pathlib import Path

import matplotlib.pyplot as plt
from tap import Tap

from src.config import base_model_names, model_display_name_dict, \
    display_name_of_downstream_evaluation_dataset_dict
from src.path import get_downstream_evaluation_metrics_path, performance_figures_dir
from src.downstream_evaluation.sample_and_rank.get_performance_and_table import \
    SampleAndRankPerformanceAndTableTap, \
    get_sample_and_rank_selected_output_path_dict, \
    analysis_dataset_list


# Define the training dataset checkpoints to compare.
dataset_size_configs = [
    ("10K", 10, "FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_10k_duplicated_40k_202512"),
    ("20K", 20, "FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_20k_duplicated_40k_202512"),
    ("40K", 40, "FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_202512"),
]
optimizer_name = "AdamW"
marker_palette = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "+"]
model_markers = {
    dataset_name: marker_palette[idx % len(marker_palette)]
    for idx, dataset_name in enumerate(analysis_dataset_list)
}


# CLI options for generating the ablation figure.
class DataSizeAblationTap(Tap):
    sample_k: int = 7
    verification_score_type: str = "logprob_min"
    evaluation_metric: str = "accuracy"
    output_format: str = "pdf"


# Build the Sample-and-Rank configuration object we reuse across loads.
def build_sample_and_rank_args(sample_k: int) -> SampleAndRankPerformanceAndTableTap:
    args = SampleAndRankPerformanceAndTableTap().parse_args([])
    args.evaluation_mode = "final_evaluation"
    args.initial_generation_prompt = "few-shot"
    args.verification_prompt_type = "multi-turn"
    args.sample_k = sample_k
    return args


# Collect accuracy curves per dataset without averaging across tasks.
def collect_dataset_curves(
    base_model: str,
    sr_args: SampleAndRankPerformanceAndTableTap,
    metric_name: str,
    verification_score_type: str,
) -> dict[str, tuple[list[int], list[float]]]:
    dataset_curves: dict[str, tuple[list[int], list[float]]] = {}

    for dataset_name in analysis_dataset_list:
        # Locate downstream evaluation outputs for each verifier.
        outputs = get_sample_and_rank_selected_output_path_dict(
            sr_args,
            base_model_name=base_model,
            initial_response_model_name=base_model,
            evaluation_dataset_name=dataset_name,
            verification_score_type=verification_score_type,
        )

        for _, size, train_data_name in dataset_size_configs:
            method_name = f"fover_{train_data_name}_{optimizer_name}"
            prediction_path = outputs.get(method_name)
            if prediction_path is None:
                raise FileNotFoundError(
                    f"Missing prediction path for {base_model}, {dataset_name}, {train_data_name}."
                )

            metrics_path = get_downstream_evaluation_metrics_path(
                dataset_name=dataset_name,
                model_name=base_model,
                prediction_path=prediction_path,
                split="test",
            )
            if not metrics_path.exists():
                raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metric_value = metrics.get(metric_name)
            if metric_value is None:
                raise ValueError(f"Metric '{metric_name}' not found in {metrics_path}")

            sizes_list, scores_list = dataset_curves.setdefault(dataset_name, ([], []))
            sizes_list.append(size)
            scores_list.append(float(metric_value) * 100.0)

    # Confirm every dataset has a value for each training size.
    expected_size_count = len(dataset_size_configs)
    for dataset_name, (sizes_list, scores_list) in dataset_curves.items():
        if len(scores_list) != expected_size_count:
            raise ValueError(
                f"Dataset {dataset_name} for model {base_model} has {len(scores_list)} values, "
                f"expected {expected_size_count}."
            )

    return dataset_curves


# Render one figure per base model showing dataset-specific curves.
def plot_dataset_curves(base_model: str, dataset_curves: dict[str, tuple[list[int], list[float]]], output_path: Path) -> None:
    plt.figure(figsize=(5, 2.5))

    model_label = model_display_name_dict.get(base_model, base_model.split("/")[-1])

    for dataset_name in analysis_dataset_list:
        if dataset_name not in dataset_curves:
            continue
        sizes, scores = dataset_curves[dataset_name]
        dataset_label = display_name_of_downstream_evaluation_dataset_dict.get(
            dataset_name, dataset_name
        )
        marker = model_markers.get(dataset_name, marker_palette[0])
        plt.plot(sizes, scores, marker=marker, label=dataset_label)

    plt.xticks([10, 20, 40], [label for label, _, _ in dataset_size_configs])
    plt.xlabel("Training Data Size")
    plt.ylabel("Best-of-K Accuracy (%)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    # two column legend, center but slightly lower
    plt.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, 0.4))
    # plt.title(f"{model_label}")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close()


# Entry point to load metrics and create the figure.
def main() -> None:
    args = DataSizeAblationTap().parse_args()
    sr_args = build_sample_and_rank_args(args.sample_k)

    collected_results: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    missing_models: list[tuple[str, str]] = []

    # Aggregate results for each base model under consideration.
    for base_model in ["meta-llama/Llama-3.1-8B-Instruct"]:  # base_model_names:
        try:
            dataset_curves = collect_dataset_curves(
                base_model,
                sr_args,
                metric_name=args.evaluation_metric,
                verification_score_type=args.verification_score_type,
            )
            collected_results[base_model] = dataset_curves
        except (FileNotFoundError, ValueError) as exc:
            missing_models.append((base_model, str(exc)))

    if missing_models:
        details = "; ".join(
            f"{model_display_name_dict.get(model, model.split('/')[-1])}: {reason}"
            for model, reason in missing_models
        )
        raise RuntimeError(
            "Missing evaluation metrics encountered. Aborting figure generation. Details: "
            + details
        )

    if not collected_results:
        raise RuntimeError("No data available to plot.")

    for base_model, dataset_curves in collected_results.items():
        model_short_name = base_model.split("/")[-1]
        figure_name = (
            f"data_size_ablation_{model_short_name}_samplek={args.sample_k}.{args.output_format}"
        )
        output_path = performance_figures_dir / "data_size_ablation" / figure_name
        plot_dataset_curves(base_model, dataset_curves, output_path)


if __name__ == "__main__":
    main()
