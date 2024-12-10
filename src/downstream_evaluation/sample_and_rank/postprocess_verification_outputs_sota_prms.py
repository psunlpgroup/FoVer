import json

import torch
import transformers

from src.path import get_verification_for_sample_and_rank_outputs_path, \
    get_verification_scores_for_sample_and_rank_path
from src.downstream_evaluation.sample_and_rank.run_verification import EvaluationForSampleAndRankTap
from src.config import sota_prms_dict, base_model_names
from src.utils.prm.postprocess import get_postprocessed_prm_output_format, \
    postprocess_prm_output_from_vllm_reward_model
from src.load_dataset import load_existing_dataset
from src.llm.utils import save_md5_hash


def get_verification_from_hidden_states_of_causal_model(
        verification_model_name: str,
        raw_verification_outputs: list[dict],
    ):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        verification_model_name
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        verification_model_name, torch_dtype="auto", device_map="auto",
    )

    # target tokens
    if verification_model_name in [
                "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
                "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
            ]:
        positive_token_id = tokenizer.encode("+", add_special_tokens=False)[0]
        negative_token_id = tokenizer.encode("-", add_special_tokens=False)[0]
    elif model.config.architectures[0] in ["LlamaForCausalLM", "Qwen2ForCausalLM"]:
        positive_token_id = tokenizer.encode("correct", add_special_tokens=False)[0]
        negative_token_id = tokenizer.encode("incorrect", add_special_tokens=False)[0]
    else:
        raise ValueError(
            f"Model {verification_model_name} is not supported for " \
            f"getting verification scores from hidden states."
        )

    # proejct features to logits
    postprocessed_output = []
    for output in raw_verification_outputs:
        features = torch.tensor(output["response"], dtype=torch.bfloat16)
        logits = model.lm_head(features)

        positive_logits = logits[:, positive_token_id]
        negative_logits = logits[:, negative_token_id]

        logits_pair = torch.stack([negative_logits, positive_logits], dim=1)
        probs = torch.nn.functional.softmax(logits_pair, dim=1).tolist()

        # get the postprocessed output format
        postprocessed_output.append({"id": output["id"], "response": probs})
    
    return postprocessed_output


def main():
    args = EvaluationForSampleAndRankTap().parse_args()
    
    # although the reward scores are not necessarily logprob, we use the same
    # names as in FoVer
    for verification_score_type in ["logprob_min"]:
        
        # save verification scores
        verification_scores_path = get_verification_scores_for_sample_and_rank_path(
            dataset_name=args.dataset_name,
            base_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            verification_score_type=verification_score_type,
            split="test",
            prompt_type=args.verification_prompt_type,
        )
        verification_scores_path.parent.mkdir(parents=True, exist_ok=True)
        
        for sample_idx in range(args.sample_k):
            # load verification outputs
            verification_outputs_path = get_verification_for_sample_and_rank_outputs_path(
                dataset_name=args.dataset_name,
                initial_response_model_name=args.initial_generation_model_name,
                verification_model_name=args.verification_model_name,
                split="test",
                prompt_type=args.verification_prompt_type,
                sample_idx=sample_idx,
            )
            with open(verification_outputs_path, "r") as f:
                raw_verification_outputs = [json.loads(line) for line in f]
            
            # if intermediate states are stored, project to logits
            if args.verification_model_name in [
                        "Qwen/Qwen2.5-Math-7B-PRM800K",
                        "Qwen/Qwen2.5-Math-PRM-7B",
                        "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
                    ]:
                # for this model, the reward scores are generated
                verification_outputs = raw_verification_outputs
            else:
                # convert to logits (reward)
                verification_outputs = get_verification_from_hidden_states_of_causal_model(
                    verification_model_name=args.verification_model_name,
                    raw_verification_outputs=raw_verification_outputs,
                )
            
            # postprocess verification outputs
            processed_verification_results = []
            for verification_output in verification_outputs:
                processed_output = postprocess_prm_output_from_vllm_reward_model(
                    verification_output=verification_output,
                    verification_model=args.verification_model_name,
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
