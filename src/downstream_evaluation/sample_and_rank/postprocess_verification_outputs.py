""" This code postprocess and evaluates the outputs of the verification model for sample-and-rank. """

import json

from src.path import get_downstream_evaluation_initial_responses_path, \
    get_verification_for_sample_and_rank_outputs_path, \
    get_verification_scores_for_sample_and_rank_path
from src.downstream_evaluation.sample_and_rank.run_verification import EvaluationForSampleAndRankTap
from src.downstream_evaluation.utils import get_solution_steps_from_response
from src.utils.prm import postprocess_prm_output
from src.llm.utils import save_md5_hash


def main():
    args = EvaluationForSampleAndRankTap().parse_args()
    
    for verification_score_type in ["logprob_min"]:
        
        # save verification scores
        verification_scores_path = get_verification_scores_for_sample_and_rank_path(
            dataset_name=args.dataset_name,
            base_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            verification_score_type=verification_score_type,
            split="test",
            prompt_type="zero-shot"
        )
        verification_scores_path.parent.mkdir(parents=True, exist_ok=True)
        
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
            
            # load verification outputs
            verification_outputs_path = get_verification_for_sample_and_rank_outputs_path(
                dataset_name=args.dataset_name,
                initial_response_model_name=args.initial_generation_model_name,
                verification_model_name=args.verification_model_name,
                split="test",
                prompt_type="zero-shot",
                sample_idx=sample_idx
            )
            with open(verification_outputs_path, "r") as f:
                verification_outputs = [json.loads(line) for line in f]
            
            # get log probabilities for verification outputs
            if "logprob" in verification_score_type:
                log_prob_path = verification_outputs_path.with_suffix(".logprobs.jsonl")
                if log_prob_path.exists():
                    with open(log_prob_path, "r") as f:
                        log_probs = [json.loads(line) for line in f]
                else:
                    log_probs = []
                
                if len(log_probs) == 0:
                    import warnings
                    warnings.warn(
                        "Log probabilities are not available for" \
                        f"{args.initial_generation_model_name} on {args.dataset_name} with " \
                        f"{args.verification_model_name} verifier. " \
                        "Skipping logprob verification score calculation."
                    )
                    break
            
            # postprocess verification outputs
            processed_verification_results = []
            for data_idx in range(len(initial_responses)):
                # create pseudo y_true for postprocess_prm_output
                # we need this to get the correct number of solution steps
                initial_response = initial_responses[data_idx]
                num_solution_steps = len(
                    get_solution_steps_from_response(initial_response["response"])
                )
                pseudo_y_true = [None] * num_solution_steps
                
                # postprocess
                verification_output = verification_outputs[data_idx]
                postprocessed_output = postprocess_prm_output(
                    verification_score_type=verification_score_type,
                    original_y_true=pseudo_y_true,
                    evaluation_output=verification_output,
                    base_model_name=args.base_model_name,
                    logprobs=log_probs[data_idx]["logprobs"] if "logprob" in verification_score_type else None,
                    remove_step_if_y_true_is_none=False,
                )
                processed_verification_results.append(postprocessed_output)
            
            # save the processed verification results
            verification_scores_path_for_this_idx = \
                verification_scores_path.with_suffix(
                    f".intermediate.idx={sample_idx}.jsonl"
                )
            
            with open(verification_scores_path_for_this_idx, "w") as f:
                for verification_score in processed_verification_results:
                    f.write(json.dumps(verification_score) + "\n")
            save_md5_hash(verification_scores_path_for_this_idx)


if __name__ == "__main__":
    main()
