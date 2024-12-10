from pathlib import Path
import re

from tap import Tap
import matplotlib.pyplot as plt

from src.typing import BASE_MODEL
from src.path import performance_figures_dir
from src.config import \
    train_dataset_names_list, train_dataset_names_list_multi_turn, \
    base_model_names, \
    model_display_name_dict, finetuned_verification_models_dict
from src.utils.model_selection import get_model_selection_performance_list


model_graph_colros: dict[BASE_MODEL, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "royalblue",
    "Qwen/Qwen2.5-7B-Instruct": "darkorange",
}

dataset_markers: dict[str, str] = {
    "orca_math": "o",
    "bbh_logical_deduction_three_objects": "^",
    "bbh_boolean_expressions": "v"
}


def extract_learning_rate_llama_factory(model_path: str) -> float | None:
    """ Extract the learning rate from the README.md file of the model. """
    
    readme_file = Path(model_path) / "README.md"
    with open(readme_file, "r") as f:
        content = f.read()
    
    match = re.search(r'learning_rate:\s*([0-9.eE+-]+)', content)
    if match:
        return float(match.group(1))
    else:
        return None


class ModelSelectionTap(Tap):
    verification_score_type: str = "logprob_min"
    verification_prompt_type: str = "multi-turn"


def main():
    args = ModelSelectionTap().parse_args()
    
    model_selection_figures_dir = performance_figures_dir / "model_selection"
    model_selection_figures_dir.mkdir(parents=True, exist_ok=True)
    
    selected_train_dataset_names_list = {
        "multi-turn": train_dataset_names_list_multi_turn,
        "zero-shot": train_dataset_names_list
    }[args.verification_prompt_type]

    for optimizer in ["AdamW"]:
        for train_data_name in selected_train_dataset_names_list:
            fig = plt.figure(figsize=(6, 5))
            
            result_added_for_at_least_one_model = False
            for base_model_name in base_model_names:
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
                
                try:
                    verifier_performance_list, verifier_performance_list_per_dataset = \
                        get_model_selection_performance_list(
                            base_model_name=base_model_name,
                            verifier_names_list=candidates_list,
                            verification_score_type=args.verification_score_type,
                            verification_prompt_type=args.verification_prompt_type,
                            metric_name="accuracy"
                        )
                except FileNotFoundError:
                    print(f"File not found for {base_model_name} {optimizer} on {train_data_name}")
                    continue
                except Exception as e:
                    raise e
                
                # extract learning rate
                # and plot the average performance
                x_list, y_list = [], []
                for model_path, y in zip(
                        candidates_list, verifier_performance_list):
                    x = extract_learning_rate_llama_factory(model_path)
                    
                    x_list.append(x)
                    y_list.append(y)
                
                plt.plot(
                    x_list, y_list,
                    label=model_display_name_dict[base_model_name],
                    color=model_graph_colros[base_model_name],
                    linewidth=2,
                )

                # plot per dataset performance
                for dataset_name, y in verifier_performance_list_per_dataset.items():
                    if dataset_name not in dataset_markers:
                        continue
                    
                    plt.plot(
                        x_list, y,
                        marker=dataset_markers[dataset_name],
                        # label=f"{model_pretty_name_dict[base_model_name]} {dataset_name}",
                        linestyle="--",
                        color=model_graph_colros[base_model_name],
                    )
                
                result_added_for_at_least_one_model = True
            
            if not result_added_for_at_least_one_model:
                print(f"No results for {optimizer} on {train_data_name}")
                continue
            
            plt.xscale("log")

            plt.xlabel("Learning Rate", fontsize=12)
            plt.ylabel("Validation Accuracy", fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout(pad=0.2)
            
            fig_path = model_selection_figures_dir / \
                f"{optimizer}_{train_data_name}_accuracy.png"
            plt.savefig(fig_path)


if __name__ == "__main__":
    main()
