""" Get performance and table for sample-and-rank evaluation """

import warnings
import json
import subprocess
from pathlib import Path

from tap import Tap
import numpy as np

from src.typing import PROMPT_TYPE, BASE_MODEL, DOWNSTREAM_EVALUATION_MODE
from src.path import get_best_sample_and_rank_output_path, \
    get_downstream_evaluation_initial_responses_path, \
    get_majority_vote_output_path, \
    get_downstream_evaluation_metrics_path, \
    downstream_evaluation_tables_dir
from src.config import \
    train_dataset_names_list, train_dataset_names_list_multi_turn, \
    base_model_names, model_display_name_dict, \
    get_downstream_evaluation_datasets_lsit, \
    get_downstream_evaluation_datasets_display_name_list, \
    sota_prms_dict, \
    finetuned_verification_models_dict
from src.utils.evaluation import accuracy_paired_bootstrap_test
from src.utils.model_selection import get_best_performance_verifier
from src.analysis.figures.generate_model_selection_performance_graph import \
    extract_learning_rate_llama_factory


class SampleAndRankPerformanceAndTableTap(Tap):
    evaluation_mode: DOWNSTREAM_EVALUATION_MODE = "final_evaluation"
    # "model_selection" or "final_evaluation"
    initial_generation_prompt: PROMPT_TYPE = "few-shot"
    # "few-shot"
    verification_prompt: PROMPT_TYPE = "multi-turn"
    # "multi-turn" is for the new prompt


def get_method_display_name(method_name: str, base_model_name: str) -> str:
    if method_name == "base_model":
        return "w/o Best-of-K"
    elif method_name == "baseline_verifier":
        return model_display_name_dict[base_model_name]
    elif "fover" in method_name:
        trian_data = method_name.split("_", 1)[-1].replace("_", "-")
        return f"{model_display_name_dict[base_model_name]}-FoVer ({trian_data})"
    elif method_name == "majority_vote":
        return "Self-Consistency (Majority-of-K)"
    elif method_name == "oracle":
        return "Oracle"
    else:
        return method_name
        # raise ValueError(f"Invalid method name: {method_name}")


def get_metric_str_with_p_value(metric: float, p_value: float,
                                p_value_threshold: float=0.05) -> str:
    if p_value < p_value_threshold:
        return f"{metric:.1f}$^*$          "
    else:
        # add phantom to align decimal points
        return f"{metric:.1f}\\phantom{{$^*$}}"


def get_sample_and_rank_selected_output_path_dict(
        args: SampleAndRankPerformanceAndTableTap,
        base_model_name: BASE_MODEL,
        initial_response_model_name: str,
        evaluation_dataset_name: str,
        verification_score_type: str) -> dict[str, Path]:

    # train dataset names
    if args.verification_prompt == "zero-shot":
        selected_train_dataset_names_list = train_dataset_names_list
    elif args.verification_prompt == "multi-turn":
        selected_train_dataset_names_list = train_dataset_names_list_multi_turn
    else:
        raise NotImplementedError(
            f"Verification prompt type {args.verification_prompt} is not " \
            "supported."
        )

    # get finetuned verification models
    if args.evaluation_mode == "model_selection":
        if base_model_name != initial_response_model_name:
            raise ValueError(
                f"Base model name {base_model_name} and initial response model "
                f"name {initial_response_model_name} should be the same for "
                "model selection."
            )

        output_path_dict = {}
        for optimizer_name, models_dict in finetuned_verification_models_dict[
                base_model_name].items():
            for train_data in selected_train_dataset_names_list:
                if train_data not in models_dict.keys():
                    continue
                
                for verification_model_name in models_dict[train_data]:
                    learning_rate = extract_learning_rate_llama_factory(
                        model_path=verification_model_name
                    )
                    
                    key = f"{train_data}_{optimizer_name}_{learning_rate}"
                    output_path_dict[key] = \
                        get_best_sample_and_rank_output_path(
                            dataset_name=evaluation_dataset_name,
                            base_model_name=base_model_name,
                            verification_model_name=verification_model_name,
                            split="test",
                            verification_prompt_type=args.verification_prompt,
                            verification_score_type=verification_score_type
                        )
        return output_path_dict
    elif args.evaluation_mode == "final_evaluation":
        output = {
            "base_model": get_downstream_evaluation_initial_responses_path(
                dataset_name=evaluation_dataset_name,
                model_name=initial_response_model_name,
                split="test",
                prompt_type=args.initial_generation_prompt,
                sample_idx=0
            ).with_suffix(".postprocessed.jsonl"),
            "baseline_verifier": get_best_sample_and_rank_output_path(
                dataset_name=evaluation_dataset_name,
                base_model_name=initial_response_model_name,
                verification_model_name=base_model_name,
                split="test",
                verification_prompt_type=args.verification_prompt,
                verification_score_type=verification_score_type
            )
        }
        
        for optimizer in ["AdamW"]:
            for train_data_name in selected_train_dataset_names_list:
                verification_model_name = get_best_performance_verifier(
                    base_model_name=base_model_name,
                    train_data_name=train_data_name,
                    optimizer=optimizer
                )
                if verification_model_name is None:
                    continue
                
                output[f"fover_{train_data_name}_{optimizer}"] = \
                    get_best_sample_and_rank_output_path(
                        dataset_name=evaluation_dataset_name,
                        base_model_name=initial_response_model_name,
                        verification_model_name=verification_model_name,
                        split="test",
                        verification_prompt_type=args.verification_prompt,
                        verification_score_type=verification_score_type
                    )
        
        # existing verifiers
        for verification_model_name in sota_prms_dict[base_model_name]:
            output[verification_model_name] = \
                get_best_sample_and_rank_output_path(
                    dataset_name=evaluation_dataset_name,
                    base_model_name=initial_response_model_name,
                    verification_model_name=verification_model_name,
                    split="test",
                    verification_prompt_type=args.verification_prompt,
                    verification_score_type=verification_score_type
                )
        
        output["majority_vote"] = get_majority_vote_output_path(
            dataset_name=evaluation_dataset_name,
            model_name=initial_response_model_name,
            split="test",
            prompt_type=args.initial_generation_prompt
        )
        
        output["oracle"] = get_best_sample_and_rank_output_path(
            dataset_name=evaluation_dataset_name,
            base_model_name=initial_response_model_name,
            verification_model_name="oracle",
            split="test",
            verification_prompt_type=args.verification_prompt,
            verification_score_type=verification_score_type
        )
        
        return output
    else:
        raise ValueError(f"Invalid evaluation_mode: {args.evaluation_mode}")


def main():
    args = SampleAndRankPerformanceAndTableTap().parse_args()
    evaluation_dataset_names_list = \
        get_downstream_evaluation_datasets_lsit(args.evaluation_mode)
    
    for base_model_name, initial_response_model_name in \
            list(zip(base_model_names, base_model_names)):
        for verification_score_type in ["logprob_min"]:
            
            # get evaluation metrics
            evaluation_metrics_dict: dict[str, dict] = {}
            evaluation_metrics_path_dict: dict[str, dict] = {}  # for logging
            for evaluation_dataset_name in evaluation_dataset_names_list:
                evaluation_metrics_dict[evaluation_dataset_name] = {}
                
                outputs_path_dict = get_sample_and_rank_selected_output_path_dict(
                    args, base_model_name=base_model_name,
                    initial_response_model_name=initial_response_model_name,
                    evaluation_dataset_name=evaluation_dataset_name,
                    verification_score_type=verification_score_type
                )
                evaluation_metrics_path_dict[evaluation_dataset_name] = \
                    {
                        method_name: str(prediction_path)
                        for method_name, prediction_path
                        in outputs_path_dict.items()
                    }
                
                for method_name, prediction_path in outputs_path_dict.items():
                    if prediction_path.exists():
                        subprocess.run(
                            [
                                "python",
                                "src/downstream_evaluation/evaluation/get_performance.py",
                                "--dataset_name", evaluation_dataset_name,
                                "--base_model_name", initial_response_model_name,
                                "--prediction_path", prediction_path
                            ]
                        )
                    else:
                        warnings.warn(
                            f"File not found: {prediction_path}. " \
                            "Skipping evaluation."
                        )

                    # load evaluation metrics
                    evaluation_metrics_path = get_downstream_evaluation_metrics_path(
                        dataset_name=evaluation_dataset_name,
                        model_name=initial_response_model_name,
                        prediction_path=prediction_path,
                        split="test"
                    )
                    if evaluation_metrics_path.exists():
                        with open(evaluation_metrics_path, "r") as f:
                            evaluation_metrics = json.load(f)
                    else:
                        warnings.warn(
                            f"File not found: {evaluation_metrics_path}. " \
                            "Setting evaluation_metrics to None."
                        )
                        evaluation_metrics = None
                    evaluation_metrics_dict[evaluation_dataset_name][method_name] \
                        = evaluation_metrics


            # bootstrap test
            bootstrap_test_result = {}
            if args.evaluation_mode == "final_evaluation":
                for method_name in list(evaluation_metrics_dict.values())[0].keys():
                    if ("fover" not in method_name) and (method_name not in sota_prms_dict[base_model_name]):
                        continue
                    
                    bootstrap_test_result[method_name] = {}
                    
                    y_correct_base_all = []
                    y_correct_fover_all = []
                    for evaluation_dataset_name in evaluation_dataset_names_list:
                        print(
                            f"Bootstrap test for {base_model_name} vs. " \
                            f"{method_name} on " \
                            f"{evaluation_dataset_name}..."
                        )
                        base_model_metrics = evaluation_metrics_dict[
                            evaluation_dataset_name]["baseline_verifier"]
                        
                        fover_metrics = evaluation_metrics_dict[
                            evaluation_dataset_name][method_name]
                        
                        if base_model_metrics is not None and \
                                fover_metrics is not None:
                            y_correct_base_model = base_model_metrics["is_correct"]
                            y_correct_fover_verifier = fover_metrics["is_correct"]
                            
                            bootstrap_test_result[method_name][evaluation_dataset_name] = \
                                accuracy_paired_bootstrap_test(
                                    y_correct_fover_verifier, y_correct_base_model
                                )

                            y_correct_base_all.extend(y_correct_base_model)
                            y_correct_fover_all.extend(y_correct_fover_verifier)
                    
                    # bootstrap test for all datasets (for average)
                    print(f"Bootstrap test for {base_model_name} on all datasets...")
                    bootstrap_test_result[method_name]["all_datasets"] = \
                        accuracy_paired_bootstrap_test(
                            y_correct_fover_all, y_correct_base_all
                        )
            
            # save directory
            sample_and_rank_table_dir = \
                downstream_evaluation_tables_dir / "sample_and_rank" \
                    / f"initial_generation={initial_response_model_name.split('/')[-1]}" \
                    / f"verification_model={base_model_name.split('/')[-1]}" \
                    / args.verification_prompt / verification_score_type
            sample_and_rank_table_dir.mkdir(exist_ok=True, parents=True)


            if args.evaluation_mode == "final_evaluation":
                # save bootstrap test results
                bootstrap_test_result_path = \
                    sample_and_rank_table_dir / "bootstrap_test.json"
                with open(bootstrap_test_result_path, "w") as f:
                    json.dump(bootstrap_test_result, f, indent=4)
            
            # generate latex table
            table = []
            first_row = [f"{'Verifier':90s}"] \
                + [f"{dname:18s}" for dname in
                    get_downstream_evaluation_datasets_display_name_list(
                        evaluation_mode=args.evaluation_mode
                    )
                   ] \
                + ["Average"]
            table.append(first_row)
            table.append(["\\midrule"])
            for metric_name in ["accuracy"]:
                for method_name in list(evaluation_metrics_dict.values())[0].keys():
                    if method_name == "majority_vote":
                        table.append(["\\midrule"])
                    
                    method_display_name = get_method_display_name(
                        method_name, base_model_name
                    )
                    row = [f"{method_display_name:90s}"]

                    # metrics
                    all_metrics_list = []
                    for evaluation_dataset_name in evaluation_dataset_names_list:
                        # does not exist
                        if evaluation_metrics_dict[
                                evaluation_dataset_name][method_name] is None:
                            row.append(f"{'':18s}")
                            continue
                        
                        metric = evaluation_metrics_dict[
                            evaluation_dataset_name][method_name][metric_name] * 100
                        all_metrics_list.append(metric)
                        
                        if method_name in bootstrap_test_result.keys():
                            metric_str = get_metric_str_with_p_value(
                                metric,
                                bootstrap_test_result[method_name][
                                    evaluation_dataset_name]["p_value"]
                            )
                        else:
                            metric_str = get_metric_str_with_p_value(
                                metric, p_value=1.0  # dummy to always get phantom
                            )
                        row.append(f"{metric_str:18s}")
                    
                    # average
                    average = np.mean(all_metrics_list).item()
                    if method_name in bootstrap_test_result.keys():
                        average_str = get_metric_str_with_p_value(
                            average,
                            bootstrap_test_result[method_name]["all_datasets"]["p_value"]
                        )
                    else:
                        average_str = get_metric_str_with_p_value(
                            average, p_value=1.0  # dummy to always get phantom
                        )
                    
                    row.append(average_str)
                    
                    table.append(row)
                    
                    # add midrule after base_model
                    if method_name in ["base_model", "baseline_verifier"]:
                        table.append(["\\midrule"])
                
                # save model
                if args.evaluation_mode == "model_selection":
                    table_path = sample_and_rank_table_dir / \
                        f"model_selection_{metric_name}.txt"
                elif args.evaluation_mode == "final_evaluation":
                    table_path = sample_and_rank_table_dir / \
                        f"final_evaluation_{metric_name}.txt"
                else:
                    raise ValueError(f"Invalid evaluation_mode: {args.evaluation_mode}")
                with open(table_path, "w") as f:
                    for row in table:
                        if len(row) == 1:  # \\midrule
                            f.write(row[0] + "\n")
                        else:
                            f.write(" & ".join(row) + " \\\\\n")
                
                # remove statistical significance marks
                with open(table_path, "r") as f:
                    table_lines = f.read()
                updated_table_lines = table_lines.replace(
                        "$^*$          ", "").replace(
                        r"\phantom{$^*$}", "").replace(
                        "              ", "")
                with open(table_path.with_suffix(".plain.tex"), "w") as f:
                    f.write(updated_table_lines)

            # save evaluation_metrics_path_dict
            evaluation_metrics_path = \
                sample_and_rank_table_dir / "evaluation_metrics_path.json"
            with open(evaluation_metrics_path, "w") as f:
                json.dump(evaluation_metrics_path_dict, f, indent=4)


if __name__ == "__main__":
    main()
