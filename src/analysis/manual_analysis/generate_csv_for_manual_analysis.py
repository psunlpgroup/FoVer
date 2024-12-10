# we compare outputs from baseline verifiers and finetuned verifiers

import json
import csv

import numpy as np

from src.typing import TRAIN_DATA_MULTI_TURN
from src.config import base_model_names
from src.path import get_verification_scores_for_sample_and_rank_path, \
    get_downstream_evaluation_initial_responses_path, \
    get_downstream_evaluation_metrics_path, \
    get_annotation_csv_path
from src.utils.model_selection import get_best_performance_verifier
from src.load_dataset import load_existing_dataset
from src.downstream_evaluation.sample_and_rank.get_performance_and_table \
    import SampleAndRankPerformanceAndTableTap, \
        get_sample_and_rank_selected_output_path_dict
from src.downstream_evaluation.utils import get_solution_steps_from_response


train_data_name: TRAIN_DATA_MULTI_TURN = \
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k"
optimizer = "AdamW"
prompt_type = "multi-turn"
verification_score_type = "logprob_min"

sample_k = 5

manual_analysis_datasets_list = [
    "gsm8k", "anli", "bbh_temporal_sequences", "bbh_word_sorting", "mmlu_pro_nomath"
]

def main():
    for base_model_name in base_model_names:
        args = SampleAndRankPerformanceAndTableTap().parse_args(
            [
                "--verification_prompt", prompt_type,
            ]
        )

        finetuned_verifier_name = get_best_performance_verifier(
            base_model_name=base_model_name,
            train_data_name=train_data_name,
            optimizer=optimizer,
        )

        for dataset_name in manual_analysis_datasets_list:
            dataset = load_existing_dataset(
                dataset_name=dataset_name,
                split="test",
            )

            output_path_dict = get_sample_and_rank_selected_output_path_dict(
                args=args,
                base_model_name=base_model_name,
                initial_response_model_name=base_model_name,
                evaluation_dataset_name=dataset_name,
                verification_score_type=verification_score_type,
            )

            # load initial responses
            initial_generation_dict = {}
            for initial_response_idx in range(sample_k):
                initial_generation_path = get_downstream_evaluation_initial_responses_path(
                    dataset_name=dataset_name,
                    model_name=base_model_name,
                    prompt_type="few-shot",
                    sample_idx=initial_response_idx,
                    split="test",
                )
                with open(initial_generation_path, "r") as f:
                    initial_responses = [
                        json.loads(line) for line in f
                    ]
                initial_generation_dict[initial_response_idx] = [
                    get_solution_steps_from_response(
                        response["response"]
                    ) for response in initial_responses
                ]

            # load verification scores
            verification_scores_dict = {}
            evalaution_metrics_dir = {}
            for verifier in [base_model_name, finetuned_verifier_name]:
                key_name = {
                    base_model_name: "baseline_verifier",
                    finetuned_verifier_name: \
                        f"fover_{train_data_name}_{optimizer}"
                }[verifier]
                prediction_path = output_path_dict[key_name]
                
                evaluation_metrics_path = \
                    get_downstream_evaluation_metrics_path(
                        dataset_name=dataset_name,
                        model_name=base_model_name,
                        prediction_path=prediction_path,
                        split="test",
                    )
                with open(evaluation_metrics_path, "r") as f:
                    evalaution_metrics_dir[verifier] = json.load(f)
                
                # load verification scores
                verification_scores_dict[verifier] = {}

                verification_scores_path = get_verification_scores_for_sample_and_rank_path(
                    dataset_name=dataset_name,
                    base_model_name=base_model_name,
                    verification_model_name=verifier,
                    verification_score_type=verification_score_type,
                    split="test",
                    prompt_type=prompt_type,
                )

                with open(verification_scores_path, "r") as f:
                    verification_scores_dict[verifier]["summary"] = [
                        json.loads(line) for line in f
                    ]
                
                # load scores for each sample
                for sample_idx in range(sample_k):
                    sample_score_path = verification_scores_path.with_suffix(
                        f".intermediate.idx={sample_idx}.jsonl"
                    )

                    with open(sample_score_path, "r") as f:
                        verification_scores_dict[verifier][sample_idx] = [
                            json.loads(line) for line in f
                        ]

            
            # find cases where
            # * baseline is correct and finetuned is wrong
            # * baseline is wrong and finetuned is correct
            flipped_cases = {"baseline_correct": [], "finetuned_correct": []}
            for case_type in ["baseline_correct", "finetuned_correct"]:
                correct_cases, wrong_cases = {
                    "baseline_correct": (
                        evalaution_metrics_dir[base_model_name],
                        evalaution_metrics_dir[finetuned_verifier_name],
                    ),
                    "finetuned_correct": (
                        evalaution_metrics_dir[finetuned_verifier_name],
                        evalaution_metrics_dir[base_model_name],
                    ),
                }[case_type]

                correct_cases_correctness = correct_cases["is_correct"]
                wrong_cases_correctness = wrong_cases["is_correct"]
                for data_idx, (is_correct_case_correct, is_wrong_case_correct) \
                        in enumerate(
                            zip(
                                correct_cases_correctness,
                                wrong_cases_correctness,
                            )
                        ):

                    if is_correct_case_correct and \
                            not is_wrong_case_correct:
                        flipped_cases[case_type].append(data_idx)
                
                # create csv file
                csv_list = [
                    ["data_idx", "id", "type", "value", "baseline", "finetuned"]
                ]

                for data_idx in flipped_cases[case_type]:
                    data = dataset[data_idx]

                    csv_list.append([data_idx, data["id"]])
                    csv_list.append(["", "", "question", data["question"]])
                    csv_list.append(["", "", "y_true", data["y_true"]])

                    baseline_solution_idx = np.argmax(
                        verification_scores_dict[base_model_name][
                            "summary"
                        ][data_idx]["verification_scores"]
                    )
                    finetuned_solution_idx = np.argmax(
                        verification_scores_dict[finetuned_verifier_name][
                            "summary"
                        ][data_idx]["verification_scores"]
                    )

                    # add initial responses
                    for solution_name, solution_idx in [
                        ("baseline", baseline_solution_idx),
                        ("finetuned", finetuned_solution_idx),
                    ]:
                        csv_list.append(["", "", f"solution selected by {solution_name}"])
                        for step_idx, step in enumerate(
                            initial_generation_dict[solution_idx][data_idx]
                        ):
                            step_scores = []
                            for verifier in [base_model_name, finetuned_verifier_name]:
                                step_scores.append(
                                    verification_scores_dict[verifier][
                                        solution_idx
                                    ][data_idx]["y_pred_step_level"][step_idx]
                                )

                            csv_list.append(
                                [
                                    "", "", f"step {step_idx}", step
                                ] + step_scores
                            )

                    csv_list.append(["", "", "annotation"])
                    csv_list.append([])
                
                model_short_name = base_model_name.split("/")[-1]
                flipped_cases_path = get_annotation_csv_path(
                    model_name=model_short_name,
                    train_data_name=train_data_name,
                    dataset_name=dataset_name,
                    case_type=case_type,
                )
                flipped_cases_path.parent.mkdir(parents=True, exist_ok=True)

                # write csv file
                with open(flipped_cases_path, "w") as f:
                    writer = csv.writer(f)
                    for row in csv_list:
                        writer.writerow(row)


if __name__ == "__main__":
    main()
