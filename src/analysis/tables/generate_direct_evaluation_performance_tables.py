""" Generate LaTeX tables for the verification performance of the models. """


import json
from pathlib import Path

import numpy as np
from tap import Tap

from src.typing import OPTIMIZERS
from src.config import (
    train_dataset_names_list, train_dataset_names_list_multi_turn,
    base_model_names, model_config_dict,
    get_direct_evaluation_datasets_list,
    get_direct_evaluation_dataset_pretty_name,
    sota_prms_list, sota_prms_dict,
    HF_ACCOUNT
)
from src.path import get_direct_evaluation_metrics_path, tables_dir
from src.utils.model_selection import get_best_performance_verifier
from src.utils.evaluation import float_scores_paired_bootstrap_test
from src.downstream_evaluation.sample_and_rank.get_performance_and_table \
    import get_metric_str_with_p_value


class DirectEvaluationTableTap(Tap):
    verification_prompt_type: str = "multi-turn"
    p_value_threshold: float = 0.05


def main():
    args = DirectEvaluationTableTap().parse_args()

    if HF_ACCOUNT is None:
        raise ValueError("HF_ACCOUNT is not set.")
    
    selected_train_dataset_names_list = {
        "multi-turn": train_dataset_names_list_multi_turn,
        "zero-shot": train_dataset_names_list,
    }[args.verification_prompt_type]

    
    for train_data_name in [None] + selected_train_dataset_names_list:
        print(f"Generating tables for {train_data_name}...")
        # remove this part later
        if train_data_name is not None:
            continue
        #

        direct_evaluation_tables_dir = tables_dir / "direct_evaluation"
        direct_optimizer_list: list[OPTIMIZERS] = [
            "AdamW",
        ]
        
        for evaluation_unit in ["step_level", "instance_level"]:
            for metric_name in ["auroc"]:  # , "f1", "precision", "recall"]:  # todo
                for dataset_type in ["in-distribution", "out-of-distribution"]:
                    print(f"Generating tables for {evaluation_unit} {metric_name} on {dataset_type}...")

                    metric_table_dir: Path = direct_evaluation_tables_dir / \
                        (train_data_name if train_data_name is not None else "none") / \
                        f"verification_prompt={args.verification_prompt_type}" \
                        / evaluation_unit / metric_name / dataset_type
                    metric_table_dir.mkdir(parents=True, exist_ok=True)

                    # make tables
                    for base_model_name in base_model_names + sota_prms_list + ["meta-llama/Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]:

                        print(f"Generating tables for {base_model_name}...")
                        
                        # improve this part later
                        if base_model_name in sota_prms_list + ["meta-llama/Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]:
                            if dataset_type == "in-distribution":
                                continue
                            if train_data_name is not None:
                                continue


                        table: list[list[str]] = []  # list of rows
                        splits_list = ["test"] if dataset_type == "out-of-distribution" else ["train", "test"]
                        
                        # select datasets
                        all_dataset_names_list = get_direct_evaluation_datasets_list(
                            base_model_name, train_data_name=train_data_name
                        )
                        
                        if dataset_type == "in-distribution":
                            evaluation_datasets_list = [
                                dataset_name for dataset_name in all_dataset_names_list
                                if "fover" in dataset_name
                            ]
                        else:
                            evaluation_datasets_list = [
                                dataset_name for dataset_name in all_dataset_names_list
                                if "fover" not in dataset_name
                            ]
                        
                        # header
                        first_row = ["Fine-tuning", "Optimizer"]
                        for dataset_name in evaluation_datasets_list:
                            if dataset_type == "out-of-distribution":
                                first_row.append(get_direct_evaluation_dataset_pretty_name(dataset_name))
                            else:
                                for split in ["train", "test"]:
                                    first_row.append(f"{get_direct_evaluation_dataset_pretty_name(dataset_name)} ({split})")
                        
                        first_row += ["Average"]
                        table.append(first_row)
                        
                        # This is an old version. To be removed later.
                        # # majority label baseline performance
                        # if metric_name != "auroc":
                        #     row = ["\\multicolumn{2}{c}{Majority Label Baseline}"]
                        #     all_metrics: list[float] = []
                        #     for dataset_name in evaluation_datasets_list:
                        #         for split in splits_list:
                        #             # evaluation metrics file for any model includes the majority label baseline performance
                        #             evaluation_metrics_path = get_direct_evaluation_metrics_path(
                        #                 dataset_name=dataset_name, base_model_name=base_model_name, verification_model_name=base_model_name, split=split, verification_prompt_type=args.verification_prompt_type
                        #             )
                                    
                        #             # add metric to row
                        #             if evaluation_metrics_path.exists():
                        #                 with open(evaluation_metrics_path, "r") as f:
                        #                     evaluation_metrics: dict[str, dict[str, dict[str, float]]] = json.load(f)
                                        
                        #                 if metric_name in evaluation_metrics["majority_label_baseline"][evaluation_unit].keys():
                        #                     metric = evaluation_metrics["majority_label_baseline"][evaluation_unit][metric_name]

                        #                     row.append(f"{metric*100:5.1f}")
                                        
                        #                     # for average calculation
                        #                     all_metrics.append(metric)
                        #                 else:
                        #                     row.append("  -- ")
                        #             else:
                        #                 row.append("  -- ")

                        #     # average
                        #     row.append(f"{np.mean(all_metrics)*100:5.1f}")
                        #     table.append(row)
                            
                        #     # midrule
                        #     table.append(["\\midrule"])
                        
                        # rows
                        for optimizer in direct_optimizer_list:

                            if base_model_name in sota_prms_list:
                                llm_without_finetuning = None
                                for key, value in sota_prms_dict.items():
                                    if base_model_name in value:
                                        llm_without_finetuning = key
                                        break
                                if llm_without_finetuning is None:
                                    raise ValueError(f"Cannot find the base model name {base_model_name} in sota_prms_dict.")

                                # baseline models
                                verification_models_list = [llm_without_finetuning, base_model_name]
                            elif train_data_name is None:
                                verification_models_list = [base_model_name]
                            else:
                                # get list of verification models
                                verification_models_list = []
                                verifier_name = get_best_performance_verifier(
                                    base_model_name=base_model_name,
                                    train_data_name=train_data_name,
                                    optimizer=optimizer,
                                )
                                if verifier_name is not None:
                                    verification_models_list.append(verifier_name)
                                
                                if optimizer == direct_optimizer_list[0]:
                                    verification_models_list = [base_model_name] + verification_models_list
                            
                            # for bootstrap
                            baseline_predictions: \
                                dict[str, dict[str, list[float]]] = {
                                    dataset_name: {
                                        split: [] for split in splits_list
                                    }
                                    for dataset_name in evaluation_datasets_list
                                }

                            for verification_model_name in verification_models_list:
                                if "fover" in verification_model_name:
                                    model_config = model_config_dict[verification_model_name]
                                else:
                                    model_config = {
                                        "finetuning": "Original Model",
                                        "optimizer": ""
                                    }
                                
                                # for average
                                all_metrics: list[float] = []

                                # for bootstrap test
                                all_y_pred_1: list[float] = []
                                all_y_pred_2: list[float] = []
                                all_y_true: list[bool] = []
                                
                                # fine-tuning settings
                                finetuning_dataset = model_config["finetuning"]
                                if len(model_config["optimizer"]) > 0:
                                    row = [f"{finetuning_dataset:60s}", f"{model_config['optimizer']:10s}"]
                                else:
                                    # original model
                                    text = f"\\multicolumn{{2}}{{c}}{{{finetuning_dataset}}}"
                                    row = [f"{text:73s}"]
                                
                                # add metric for each evaluation dataset to row
                                for split in splits_list:
                                    for dataset_name in evaluation_datasets_list:
                                        # load evaluation metrics
                                        evaluation_metrics_path = get_direct_evaluation_metrics_path(
                                            dataset_name=dataset_name,
                                            verification_model_name=verification_model_name, split=split, verification_prompt_type=args.verification_prompt_type
                                        )
                                        
                                        # add metric to row
                                        if evaluation_metrics_path.exists():
                                            with open(evaluation_metrics_path, "r") as f:
                                                evaluation_metrics = json.load(f)
                                            
                                            if evaluation_metrics["performance"][evaluation_unit] is None:
                                                print(f"Performance metrics not found for {verification_model_name} on {dataset_name}.")
                                                row.append("  -- ")
                                                continue
                                            
                                            if metric_name in evaluation_metrics["performance"][evaluation_unit].keys():
                                                # bootstrap test
                                                p_value = 1.0
                                                if verification_model_name == verification_models_list[0]:
                                                    baseline_predictions[dataset_name][split] = evaluation_metrics[f"y_pred_{evaluation_unit}"]
                                                else:
                                                    y_pred_1 = evaluation_metrics[f"y_pred_{evaluation_unit}"]
                                                    y_pred_2 = baseline_predictions[dataset_name][split]
                                                    y_true = evaluation_metrics[f"y_true_{evaluation_unit}"]

                                                    if metric_name == "auroc":
                                                        threshold = None
                                                    else:
                                                        threshold = evaluation_metrics["threshold"]

                                                    p_value = float_scores_paired_bootstrap_test(
                                                        y_pred_1=y_pred_1, y_pred_2=y_pred_2, y_true=y_true,
                                                        metric_name=metric_name, threshold=threshold,
                                                        sample_num=1000, seed=68
                                                    )["p_value"]

                                                    all_y_pred_1 += y_pred_1
                                                    all_y_pred_2 += y_pred_2
                                                    all_y_true += y_true

                                                # get metric
                                                metric = evaluation_metrics["performance"][evaluation_unit][metric_name]
                                                row.append(
                                                    get_metric_str_with_p_value(
                                                        metric=metric*100, p_value=p_value,
                                                        p_value_threshold=args.p_value_threshold
                                                    )
                                                )

                                                # for average calculation
                                                all_metrics.append(metric)
                                            else:
                                                row.append("  -- ")
                                        else:
                                            row.append("  -- ")
                                
                                # p-value for average
                                if len(all_y_pred_1) == 0 or len(all_y_pred_2) == 0:
                                    print(f"y_pred_1 or y_pred_2 is empty for {verification_model_name} on {dataset_name}.")
                                    p_value = 1.0
                                elif verification_model_name == verification_models_list[0]:
                                    p_value = 1.0
                                else:
                                    if metric_name == "auroc":
                                        p_value = float_scores_paired_bootstrap_test(
                                            y_pred_1=all_y_pred_1, y_pred_2=all_y_pred_2,
                                            y_true=all_y_true,
                                            metric_name=metric_name,
                                            sample_num=1000, seed=68
                                        )["p_value"]
                                    else:
                                        p_value = 1.0  # update this later

                                # average
                                average = np.mean(all_metrics).item()*100 if len(all_metrics) > 0 else 0
                                row.append(
                                    get_metric_str_with_p_value(
                                        metric=average, p_value=p_value,
                                        p_value_threshold=args.p_value_threshold
                                    )
                                )
                                table.append(row)
                        
                        # save table
                        table_path = metric_table_dir / f"{base_model_name.split('/')[-1]}.tex"
                        with open(table_path, "w") as f:
                            for row in table:
                                if len(row) > 1:
                                    f.write(" & ".join(row) + " \\\\\n")
                                else:
                                    # midrule
                                    f.write(row[0] + "\n")


if __name__ == "__main__":
    main()
