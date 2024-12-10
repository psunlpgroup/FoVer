from pathlib import Path

import matplotlib.pyplot as plt

from src.path import performance_figures_dir
from src.config import \
    base_model_names, \
    model_display_name_dict, finetuned_verification_models_dict, \
    display_name_of_downstream_evaluation_dataset_dict
from src.utils.model_selection import get_model_selection_performance_list
from src.analysis.figures.generate_model_selection_performance_graph import \
    ModelSelectionTap, model_graph_colros, dataset_markers

optimizer = "AdamW"

    
selected_train_dataset_names_dict = {
    "dataset_size": [
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k",
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k",
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k",
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_duplicated_40k",
    ],
    "label_distribution": [
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25",
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.50",
        "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75",
    ]
}

ablation_parameters_dict = {
    "dataset_size": [5000, 10000, 20000, 40000],
    "label_distribution": [0.25, 0.5, 0.75]
}

x_label_dict = {
    "dataset_size": "Dataset Size",
    "label_distribution": "Label Distribution (Positive Ratio)"
}


def main():
    args = ModelSelectionTap().parse_args()
    
    ablation_study_figures_dir = performance_figures_dir / "ablation_study"
    ablation_study_figures_dir.mkdir(parents=True, exist_ok=True)

    for evaluation_category in ["dataset_size", "label_distribution"]:
        selected_train_dataset_names_list = selected_train_dataset_names_dict[
            evaluation_category
        ]

        fig = plt.figure(figsize=(6, 5))
        for base_model_name in base_model_names:
            result_added_for_at_least_one_model = False
            
            verifier_performance_list = []
            verifier_performance_list_per_dataset: dict[str, list[float]] = {}
            for train_data_name in selected_train_dataset_names_list:
                if train_data_name not in finetuned_verification_models_dict[
                    base_model_name][optimizer]:
                    print(f"No verifiers for {base_model_name} {optimizer} on "
                          f"{train_data_name}")
                    continue

                candidates_list = finetuned_verification_models_dict[
                    base_model_name][optimizer][train_data_name]

                if len(candidates_list) == 0:
                    print(f"No verifiers for {base_model_name} {optimizer} on {train_data_name}")
                    continue

                # in this code we assume that there is only one verifier for each
                if len(candidates_list) > 1:
                    raise ValueError(
                        f"More than one verifier for {base_model_name} {optimizer} on "
                        f"{train_data_name}"
                    )
                
                try:
                    verifier_performance, verifier_performance_per_dataset = \
                        get_model_selection_performance_list(
                            base_model_name=base_model_name,
                            verifier_names_list=candidates_list,
                            verification_score_type=args.verification_score_type,
                            verification_prompt_type=args.verification_prompt_type,
                            metric_name="accuracy"
                        )
                    verifier_performance_list.append(verifier_performance[0])

                    for key, values in verifier_performance_per_dataset.items():
                        verifier_performance_list_per_dataset.setdefault(
                            key, []
                        ).append(values[0])

                except FileNotFoundError:
                    print(f"File not found for {base_model_name} {optimizer} on {train_data_name}")
                    continue
                except Exception as e:
                    raise e
            
            # PLOT

            # average
            x_list, y_list = [], []
            for idx, y in enumerate(verifier_performance_list):
                x = ablation_parameters_dict[evaluation_category][idx]
                
                x_list.append(x)
                y_list.append(y)
            
            plt.plot(
                x_list, y_list,
                label=model_display_name_dict[base_model_name],
                color=model_graph_colros[base_model_name],
                linewidth=2,
            )

            # per dataset
            for dataset_name in verifier_performance_list_per_dataset.keys():
                x_list, y_list = [], []
                for idx, y in enumerate(verifier_performance_list_per_dataset[dataset_name]):
                    x = ablation_parameters_dict[evaluation_category][idx]
                    
                    x_list.append(x)
                    y_list.append(y)
                
                plt.plot(
                    x_list, y_list,
                    # label=f"{model_pretty_name_dict[base_model_name]} ({display_name_of_downstream_evaluation_dataset_dict[dataset_name]})",
                    color=model_graph_colros[base_model_name],
                    linestyle="--",
                    marker=dataset_markers[dataset_name],
                )

            
            result_added_for_at_least_one_model = True
            
            if not result_added_for_at_least_one_model:
                print(f"No results for {optimizer} on {base_model_name}")
                continue
            
            if evaluation_category == "dataset_size":
                plt.xscale("log")

            plt.xlabel(x_label_dict[evaluation_category], fontsize=12)
            plt.ylabel("Validation Accuracy", fontsize=12)
            # plt.legend(fontsize=12)
            plt.tight_layout(pad=0.2)
            
            fig_path = ablation_study_figures_dir / \
                f"{evaluation_category}_accuracy.png"
            plt.savefig(fig_path)


if __name__ == "__main__":
    main()
