# postprocess annotation and generate bar chart

import csv

import numpy as np
import matplotlib.pyplot as plt

from src.typing import TRAIN_DATA_MULTI_TURN
from src.config import base_model_names, model_display_name_dict, \
    display_name_of_downstream_evaluation_dataset_dict
from src.path import get_annotation_csv_path, manual_analysis_dir
from src.analysis.manual_analysis.generate_csv_for_manual_analysis import \
    manual_analysis_datasets_list


train_data_name: TRAIN_DATA_MULTI_TURN = \
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k"

annotation_label_candidates = [
    "baseline_wrong",
    "wrong_ground_truth", "wrong_intermediate",
    "both_wrong_scores", "finetuned_wrong"
]

label_colors = {
    "baseline_wrong": "skyblue",
    "wrong_ground_truth": "gray",
    "wrong_intermediate": "darkgray",
    "both_wrong_scores": "silver",
    "finetuned_wrong": "orangered",
}

label_hatches = {
    "baseline_wrong":          "",
    "wrong_ground_truth":      "/",    # slashes
    "wrong_intermediate":      ".",    # dotted
    "both_wrong_scores":       "x",    # crosshatch
    "finetuned_wrong":         "",
}


def main():
    proportions_dict: dict[str, dict[str, dict[str, float]]] = {}

    for correct_model in ["baseline", "finetuned"]:
        case_type = f"{correct_model}_correct"


        for base_model_name in base_model_names:
            proportions_dict[base_model_name] = {}

            for evaluation_dataset_name in manual_analysis_datasets_list:

                annotation_csv_path = get_annotation_csv_path(
                    model_name=base_model_name,
                    train_data_name=train_data_name,
                    dataset_name=evaluation_dataset_name,
                    case_type=case_type,
                    annotated=True,
                )
                
                # load the csv file
                annotated_labels = []
                with open(annotation_csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    data = list(reader)

                    # Extract the relevant columns
                    # if the third column is "annotation", the forth column includes
                    # the annotated label
                    for row in data:
                        if row["type"] == "annotation":
                            if row["value"] in annotation_label_candidates:
                                # check if the label is in the candidates
                                # and append to the list
                                annotated_labels.append(row["value"])
                            else:
                                raise ValueError(
                                    f"Invalid label: '{row['value']}' in {annotation_csv_path}. "
                                    f"Expected one of {annotation_label_candidates}"
                                )
                    
                # Count the occurrences of each label
                label_counts = {
                    label: annotated_labels.count(label) for
                    label in annotation_label_candidates
                }

                label_proportions = {
                    label: count / len(annotated_labels) * 100
                    for label, count in label_counts.items()
                }

                proportions_dict[base_model_name][
                    evaluation_dataset_name] = label_proportions

        # --- plotting (100%‐stacked single bar per subplot) ---
        models = base_model_names
        datasets = manual_analysis_datasets_list
        n_models, n_dsets = len(models), len(datasets)

        fig, axes = plt.subplots(n_models, n_dsets,
                                figsize=(n_dsets * 4, n_models * .7),
                                sharex='col', sharey='row')

        # ensure axes is 2D
        if n_models == 1:
            axes = axes[np.newaxis, :]
        if n_dsets == 1:
            axes = axes[:, np.newaxis]

        for i, model in enumerate(models):
            for j, dset in enumerate(datasets):
                ax = axes[i, j]
                vals = [proportions_dict[model][dset][lbl] for lbl in annotation_label_candidates]
                cols = [label_colors[lbl] for lbl in annotation_label_candidates]
                hatches = [label_hatches[lbl] for lbl in annotation_label_candidates]

                # draw stacked bar
                left = 0.0
                for v, c, hatch in zip(vals, cols, hatches):
                    ax.barh(0, v, left=left, height=0.6, color=c, hatch=hatch, edgecolor='white' if hatch else None,)
                    if v >= 1:
                        ax.text(left + v/2, 0, f"{int(v)}%", va="center", ha="center", fontsize=12)
                    left += v

                ax.set_xlim(0, 100)
                ax.set_yticks([])

                # remove surrounding box (all spines) :contentReference[oaicite:0]{index=0}
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # remove x‐axis ticks :contentReference[oaicite:1]{index=1}
                ax.set_xticks([])

                # model name on left
                if j == 0:
                    ax.set_ylabel(model_display_name_dict[model],
                                rotation=0, labelpad=50, va="center", fontsize=12)
                # dataset name on bottom
                if i == n_models - 1:
                    ax.set_xlabel(display_name_of_downstream_evaluation_dataset_dict[dset],
                                rotation=0, labelpad=10, fontsize=12)

        plt.tight_layout()
        figure_path = manual_analysis_dir / "figures" / \
                    f"annotation_proportions_{train_data_name}_{case_type}.png"

        plt.savefig(figure_path, dpi=300)


if __name__ == "__main__":
    main()
