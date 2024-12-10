import json

import numpy as np
import matplotlib.pyplot as plt

from src.config import base_model_names, \
    downstream_evaluation_datasets_list, \
    downstream_evaluation_dataset_name_to_category_dict
from src.path import get_downstream_evaluation_metrics_path, \
    performance_figures_dir
from src.downstream_evaluation.sample_and_rank.get_performance_and_table \
    import SampleAndRankPerformanceAndTableTap, \
        get_sample_and_rank_selected_output_path_dict


categories = ["math", "logic", "nli", "mmlu", "bbh"]
categories_display_names = [
    "Math\nReasoning", "Logical\nReasoning", "NLI", "MMLU-Pro\nNoMath",
    "BBH\n(3 tasks)"
]

methods_display_name = ["Original", "Ours"]

bar_colors = ["darkgray", "blueviolet"]

def main():
    args = SampleAndRankPerformanceAndTableTap().parse_args(
        [
            "--evaluation_mode", "final_evaluation",
            "--initial_generation_prompt", "few-shot",
            "--verification_prompt", "multi-turn"
        ]
    )
    verification_score_type = "logprob_min"
    evaluation_metric_name = "accuracy"

    for train_data_name in [
                # "fldx2_symbol_multi_turn_10k",
                # "isabelle_all_multi_turn_10k",
                # "fldx2_symbol-isabelle_all_multi_turn_10k",
                # "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k",
                "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k"
            ]:
        print("Using train data name:", train_data_name)

        methods = ["baseline_verifier", f"fover_{train_data_name}_AdamW"]

        # load downstream evaluation performance
        for base_model_name in base_model_names:
            category_performance_dict: dict[str, dict[str, list[float]]] = {}

            for evaluation_dataset_name in downstream_evaluation_datasets_list:
                category = downstream_evaluation_dataset_name_to_category_dict[
                    evaluation_dataset_name]
                
                outputs_path_dict = get_sample_and_rank_selected_output_path_dict(
                    args, base_model_name=base_model_name,
                    evaluation_dataset_name=evaluation_dataset_name,
                    verification_score_type=verification_score_type
                )

                for method_name in methods:
                    prediction_path = outputs_path_dict[method_name]
                    
                    evaluation_metrics_path = get_downstream_evaluation_metrics_path(
                        dataset_name=evaluation_dataset_name,
                        model_name=base_model_name,
                        prediction_path=prediction_path,
                        split="test"
                    )
                    if evaluation_metrics_path.exists():
                        with open(evaluation_metrics_path, "r") as f:
                            evaluation_metrics = json.load(f)
                    else:
                        print(f"Evaluation metrics file not found for " \
                            f"{base_model_name} on " \
                            f"{evaluation_dataset_name}. Skipping...")
                        continue
                    
                    # collect the performance for each category
                    category_performance_dict.setdefault(method_name, {}
                    ).setdefault(category, []
                    ).append(evaluation_metrics[evaluation_metric_name])

            # create bar graph

            # ── 2. compute average performance per (method,category) ───────────────────────
            # result: means[method] = [mean on math, mean on logic, mean on bbh]
            means = {
                m: [np.mean(category_performance_dict[m].get(cat, [])) * 100 for cat in categories]
                for m in methods
            }

            # ── 3. set up bar positions ─────────────────────────────────────────────────────
            n_cat    = len(categories)
            n_meth   = len(methods)
            bar_w    = 0.8 / n_meth              # total group width = 0.8
            x_groups = np.arange(n_cat)          # one group per category

            # for each method, bars will sit at x_groups + offset
            offsets = np.arange(n_meth) * bar_w

            # ── 4. plot ─────────────────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 3))

            for i, m in enumerate(methods):
                bars = ax.bar(x_groups + offsets[i],
                    means[m],
                    width=bar_w,
                    label=m,
                    color=bar_colors[i],
                )

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 1.5,  # slightly above bar
                        f"{height:.1f}",
                        ha='center',
                        va='bottom',
                        fontsize=19
                    )
            
            ax.set_ylim(40, 88)

            # ── 5. two-row x-axis labels ────────────────────────────────────────────────────
            # major ticks: centered under each category-group
            group_centers = x_groups + bar_w*(n_meth-1)/2
            ax.set_xticks(group_centers)
            ax.set_xticklabels(categories_display_names, fontsize=19)
            ax.tick_params(axis='x', which='major', pad=24)

            # minor ticks: one per bar, label = method name
            # flatten all bar positions and corresponding method labels
            bar_positions = []
            bar_labels    = []
            for j, cat in enumerate(categories):
                for m in methods_display_name:
                    bar_positions.append(j + methods_display_name.index(m)*bar_w)
                    bar_labels.append(m)

            ax.set_xticks(bar_positions, minor=True)
            ax.set_xticklabels(bar_labels, minor=True, rotation=0, fontsize=14)
            ax.tick_params(axis='x', which='minor', pad=4)

            # ── 6. polish ───────────────────────────────────────────────────────────────────
            # ax.set_ylabel("Accuracy", fontsize=14)
            # ax.set_title("Per-category performance by method")
            ax.legend().set_visible(False)   # legend not needed: methods shown on x-axis
            plt.tight_layout(pad=-1)

            model_short_name = base_model_name.split("/")[-1]
            figure_path = performance_figures_dir / "downstream_evaluation" / \
                f"downstream_evaluation_performance_{model_short_name}_{train_data_name}.pdf"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(figure_path, bbox_inches='tight')
            print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
