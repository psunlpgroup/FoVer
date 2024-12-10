# oracle verification returns 1 to the correct answer and 0 to the rest


import json

from src.path import get_verification_scores_for_sample_and_rank_path, \
    get_downstream_evaluation_initial_responses_path
from src.downstream_evaluation.sample_and_rank.run_verification import EvaluationForSampleAndRankTap
from src.utils.prm.postprocess import get_postprocessed_prm_output_format
from src.llm.utils import save_md5_hash
from src.downstream_evaluation.evaluation.utils import \
    get_performance_for_downstream_evaluation


def postprocess_sota_verifier_output(
        verification_output: dict, verification_model: str) -> list[float]:
    
    if verification_model in [
                "Qwen/Qwen2.5-Math-7B-PRM800K", "Qwen/Qwen2.5-Math-PRM-7B",
                "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
                "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
            ]:
        classification_output = verification_output["response"]
        
        # the second score is for the "correct" class
        return [
            s[1] for s in classification_output
        ]
    
    elif verification_model == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
        return verification_output["response"]
    
    else:
        raise ValueError(
            f"Invalid verification model: {verification_model}"
        )


def main():
    args = EvaluationForSampleAndRankTap().parse_args()

    if args.verification_model_name != "oracle":
        raise ValueError(
            "This script is only for oracle verification. "\
            "Please use the baseline verifier script for other models."
        )
    
    # although the reward scores are not logprob, we use the same
    # names as in FoVer
    for verification_score_type in ["logprob_min"]:
        
        # save path for verification scores
        verification_scores_path = get_verification_scores_for_sample_and_rank_path(
            dataset_name=args.dataset_name,
            base_model_name=args.base_model_name,
            verification_model_name=args.verification_model_name,
            verification_score_type=verification_score_type,
            split="test",
            prompt_type=args.verification_prompt_type
        )
        verification_scores_path.parent.mkdir(parents=True, exist_ok=True)
        
        for sample_idx in range(args.sample_k):
            # load initial responses
            initial_answers_path = get_downstream_evaluation_initial_responses_path(
                dataset_name=args.dataset_name, model_name=args.base_model_name,
                split="test",
                prompt_type="few-shot",  # initial responses are always few-shot
                sample_idx=sample_idx
            ).with_suffix(".postprocessed.jsonl")
            with open(initial_answers_path, "r") as f:
                predictions_list = [json.loads(line) for line in f]
            
            # get is_correct
            performance_dict = get_performance_for_downstream_evaluation(
                dataset_name=args.dataset_name,
                predictions=predictions_list,
            )
            
            # postprocess verification outputs
            processed_verification_results = []
            for is_correct in performance_dict["is_correct"]:
                processed_output = get_postprocessed_prm_output_format(
                    y_pred_step_level=[int(is_correct)],
                    y_true_step_level=[None],
                    verification_score_type=verification_score_type,
                )
                
                processed_verification_results.append(processed_output)
            
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
