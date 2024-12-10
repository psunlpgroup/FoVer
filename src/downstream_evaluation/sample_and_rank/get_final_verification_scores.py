""" This code postprocess and evaluates the outputs of the verification model for sample-and-rank. """

import json

import numpy as np

from src.config import base_model_names, \
    downstream_evaluation_for_model_selection_datasets_list
from src.path import get_downstream_evaluation_initial_responses_path, \
    get_verification_scores_for_sample_and_rank_path, \
    get_best_sample_and_rank_output_path
from src.load_dataset import load_existing_dataset
from src.downstream_evaluation.sample_and_rank.run_verification import EvaluationForSampleAndRankTap
from src.llm.utils import save_md5_hash


def main():
    args = EvaluationForSampleAndRankTap().parse_args()
    
    dataset = load_existing_dataset(args.dataset_name)
    
    for verification_score_type in ["logprob_min"]:
        
        # if the verification model is not a base model, we filter out
        # cases where the baseline model generates invalid format
        # (i.e., does not generate "<step_1>correct</step_1>" format)
        # to make sure that we fairly compare the verification performances
        # by removing the influence of the mistakes in instruction following
        if args.verification_model_name in base_model_names:
            filtering_by_baseline_model = False
        elif args.dataset_name in downstream_evaluation_for_model_selection_datasets_list:
            # for model selection, we don't compare with the baseline model
            # so we don't filter out the cases where the baseline model
            filtering_by_baseline_model = False
        else:
            filtering_by_baseline_model = True

            baseline_verification_scores_path = get_verification_scores_for_sample_and_rank_path(
                dataset_name=args.dataset_name,
                base_model_name=args.initial_generation_model_name,
                verification_model_name=args.base_model_name,  # use the base model as verifier
                verification_score_type=verification_score_type,
                split="test",
                prompt_type=args.verification_prompt_type,
            )
            with open(baseline_verification_scores_path, "r") as f:
                baseline_verification_scores = [
                    json.loads(line) for line in f
                ]
        
        best_sample_and_rank_output_path = get_best_sample_and_rank_output_path(
            dataset_name=args.dataset_name,
            base_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            verification_prompt_type=args.verification_prompt_type,
            verification_score_type=verification_score_type,
            split="test",
        )
        
        # load the intermediate scores and save the final verification scores
        verification_scores_path = get_verification_scores_for_sample_and_rank_path(
            dataset_name=args.dataset_name,
            base_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            verification_score_type=verification_score_type,
            split="test",
            prompt_type=args.verification_prompt_type,
        )
        
        sample_idx_initial_answers: dict[int, list] = {}
        verification_scores_list: list[dict[str, list]] = []
        for sample_idx in range(args.sample_k):
            # load initial responses (to get the number of solution steps)
            initial_responses_path = get_downstream_evaluation_initial_responses_path(
                dataset_name=args.dataset_name,
                model_name=args.initial_generation_model_name,
                split="test",
                prompt_type="few-shot",
                sample_idx=sample_idx,
            )
            with open(initial_responses_path, "r") as f:
                initial_responses = [json.loads(line) for line in f]
            
            # load postprocessed initial responses (to get final answers)
            with open(initial_responses_path.with_suffix(".postprocessed.jsonl"), "r") as f:
                initial_answers = [json.loads(line) for line in f]
            sample_idx_initial_answers[sample_idx] = initial_answers
            
            
            # load intermediate verification outputs
            verification_scores_path_for_this_idx = \
                verification_scores_path.with_suffix(
                    f".intermediate.idx={sample_idx}.jsonl"
                )
            with open(verification_scores_path_for_this_idx, "r") as f:
                processed_verification_results = [
                    json.loads(line) for line in f
                ]
            
            # get verification scores
            for data_idx in range(len(initial_responses)):
                raw_verification_score = processed_verification_results[
                    data_idx]["y_pred_instance_level"]
                
                is_all_step_scores_valid = processed_verification_results[
                    data_idx]["is_all_step_scores_valid"]
                is_all_step_scores_invalid = processed_verification_results[
                    data_idx]["is_all_step_scores_invalid"]
                invalid_steps_ratio = processed_verification_results[
                    data_idx]["invalid_steps_ratio"]
                
                # processed verification score
                if filtering_by_baseline_model:
                    # if the verification score of the baseline model is invalid
                    # we set the verification score to -1000000000 and do not
                    # use this output for best-of-k
                    if not baseline_verification_scores[data_idx]["is_all_step_scores_valid"][sample_idx]:
                        verification_score = -1000000000
                    else:
                        verification_score = raw_verification_score
                else:
                    verification_score = raw_verification_score
                
                if not is_all_step_scores_valid:
                    verification_score = -1000000000

                # create a new entry or append
                if len(verification_scores_list) <= data_idx:
                    # create a new entry
                    verification_scores_list.append(
                        {
                            "id": initial_responses[data_idx]["id"],
                            "y_true": dataset[data_idx]["y_true"],
                            "verification_scores": [verification_score],
                            "raw_verification_scores": [raw_verification_score],
                            "is_all_step_scores_valid": [is_all_step_scores_valid],
                            "is_all_step_scores_invalid": [is_all_step_scores_invalid],
                            "invalid_steps_ratio": [invalid_steps_ratio],
                            "initial_responses": [
                                initial_answers[data_idx]["response"]
                            ],
                        }
                    )
                else:
                    verification_scores_list[data_idx]["verification_scores"].append(
                        verification_score
                    )
                    verification_scores_list[data_idx]["raw_verification_scores"].append(
                        raw_verification_score
                    )
                    
                    verification_scores_list[data_idx]["is_all_step_scores_valid"].append(
                        is_all_step_scores_valid
                    )
                    verification_scores_list[data_idx]["is_all_step_scores_invalid"].append(
                        is_all_step_scores_invalid
                    )
                    verification_scores_list[data_idx]["invalid_steps_ratio"].append(
                        invalid_steps_ratio
                    )
                    
                    verification_scores_list[data_idx]["initial_responses"].append(
                        initial_answers[data_idx]["response"]
                    )
        
        # check error
        if len(verification_scores_list) != len(dataset):
            import warnings
            warnings.warn(
                f"Verification scores for {args.initial_generation_model_name} on " \
                f"{args.dataset_name} with {args.verification_model_name} " \
                "verifier are not available for all instances. " \
                "Skipping verification score calculation."
            )
            continue
        
        # save the verification scores
        with open(verification_scores_path, "w") as f:
            for verification_score in verification_scores_list:
                f.write(json.dumps(verification_score) + "\n")
        save_md5_hash(verification_scores_path)
        
        # stats
        all_step_scores_valid_ratio = np.mean(
            [
                np.mean(
                    verification_scores["is_all_step_scores_valid"]
                )
                for verification_scores in verification_scores_list
            ]
        )
        all_step_scores_invalid_ratio = np.mean(
            [
                np.mean(
                    verification_scores["is_all_step_scores_invalid"]
                )
                for verification_scores in verification_scores_list
            ]
        )
        average_invalid_steps_ratio = np.mean(
            [
                np.mean(
                    verification_scores["invalid_steps_ratio"]
                )
                for verification_scores in verification_scores_list
            ]
        )
        stats = {
            "all_step_scores_valid_ratio": all_step_scores_valid_ratio,
            "all_step_scores_invalid_ratio": all_step_scores_invalid_ratio,
            "average_invalid_steps_ratio": average_invalid_steps_ratio,
        }
        stats_path = verification_scores_path.with_suffix(".stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        # select the best output based on the verification scores
        best_outputs = []
        for data_idx in range(len(initial_responses)):
            best_sample_idx = np.argmax(
                [
                    score if score is not None else -1000000000 for score in
                    verification_scores_list[data_idx]["verification_scores"]
                ]
            )
            best_verification_output = sample_idx_initial_answers[
                best_sample_idx][data_idx]
            best_outputs.append(best_verification_output)
        
        # save the best verification output
        best_sample_and_rank_output_path = get_best_sample_and_rank_output_path(
            dataset_name=args.dataset_name,
            base_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            verification_prompt_type=args.verification_prompt_type,
            verification_score_type=verification_score_type,
            split="test",
        )
        best_sample_and_rank_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_sample_and_rank_output_path, "w") as f:
            for best_output in best_outputs:
                f.write(json.dumps(best_output) + "\n")
        save_md5_hash(best_sample_and_rank_output_path)


if __name__ == "__main__":
    main()
