""" This code generates initial answers from LLMs on base datasets. We will automatically verify the outputs to create the datasets with error labels. """

import subprocess
import json
from tqdm import tqdm

from src.typing import BASE_MODEL
from src.config import base_model_names, sota_prms_list
from src.path import get_downstream_evaluation_initial_responses_path, \
    get_prompt_for_verification_for_sample_and_rank_path, \
    get_verification_for_sample_and_rank_outputs_path, \
    get_prompt_for_verification_for_sample_and_rank_by_sota_prms_path
from src.direct_evaluation.run_direct_evaluation import DirectEvaluationTap
from src.load_dataset import load_existing_dataset
from src.dataset_creation.prompts import get_verification_prompt_for_single_turn_data
from src.dataset_creation.prompts import get_user_message
from src.downstream_evaluation.utils import get_solution_steps_from_response
from src.llm.utils import save_md5_hash
from src.utils.sota_prms import get_verification_prompt_for_sota_prms


class EvaluationForSampleAndRankTap(DirectEvaluationTap):
    initial_generation_model_name: BASE_MODEL
    sample_k: int=5  # number of samples for sample-and-rank


def main():
    args = EvaluationForSampleAndRankTap().parse_args()
    print(args)
    
    base_dataset = load_existing_dataset(args.dataset_name)

    for sample_idx in range(args.sample_k):
        ###
        # create and save prompts
        print(f"Generating prompts for evaluation...")
        
        # load initial responses
        initial_responses_path = get_downstream_evaluation_initial_responses_path(
            dataset_name=args.dataset_name,
            model_name=args.initial_generation_model_name,
            split="test",
            prompt_type="few-shot",  # initial responses are always few-shot
            sample_idx=sample_idx
        )
        with open(initial_responses_path, "r") as f:
            initial_responses = [json.loads(line) for line in f]
        
        print(f"Loaded {len(initial_responses)} initial responses from {initial_responses_path}")
        
        # create prompts
        prompts: list[dict] = []
        if args.verification_model_name in sota_prms_list:
            if args.not_use_vllm_reward_task:
                raise ValueError(
                    "Baseline verifiers should use vllm reward task. "\
                    "remove --not_use_vllm_reward_task flag."
                )
            
            # baseline models
            for idx, d in tqdm(enumerate(initial_responses), total=len(initial_responses)):
                prompt_conversation = \
                    get_verification_prompt_for_sota_prms(
                        problem=base_dataset[idx]["question"],
                        solution_steps=get_solution_steps_from_response(d["response"]),
                        model_name=args.verification_model_name
                    )
                
                prompts.append(
                    {"id": d["id"], "prompt": prompt_conversation}
                )
            
            # get path for saving prompts
            prompt_for_evaluation_path = get_prompt_for_verification_for_sample_and_rank_by_sota_prms_path(
                dataset_name=args.dataset_name, model_name=args.initial_generation_model_name,
                split="test", prompt_type=args.verification_prompt_type, sample_idx=sample_idx,
                verification_model_name=args.verification_model_name
            )
        
        else:
            if args.verification_prompt_type == "multi-turn":
                if args.not_use_vllm_reward_task:
                    raise ValueError(
                        "Multi-turn verification should use vllm reward task. "\
                        "Remove --not_use_vllm_reward_task flag."
                    )
                
                # new version of our models
                from src.prm.preprocessing import get_fover_input_format
                
                for idx, d in tqdm(enumerate(initial_responses), total=len(initial_responses)):
                    prompt_conversation = get_fover_input_format(
                        problem=base_dataset[idx]["question"],
                        solution_steps=get_solution_steps_from_response(d["response"]),
                        reference_error_labels=None,  # this is a dummy labels we use for inference
                        model_role_name="assistant" if "gemma" not in args.verification_model_name else "model",
                    )
                    
                    prompts.append(
                        {"id": d["id"], "prompt": prompt_conversation}
                    )
            elif args.verification_prompt_type == "zero-shot":
                raise ValueError("This is an old version. Please use multi-turn prompt.")

                if not args.not_use_vllm_reward_task:
                    raise ValueError(
                        "Zero-shot verification should not use vllm reward task. "\
                        "Use --not_use_vllm_reward_task flag."
                    )
                
                # old version of our models
                for idx, d in enumerate(initial_responses):
                    base_d = base_dataset[idx]
                    assert base_d["id"] == d["id"], \
                        f"ID mismatch: {base_d['id']} != {d['id']}"
                    
                    prompt = get_verification_prompt_for_single_turn_data(
                        data_id=f"{d['id']}-{sample_idx}",
                        problem=base_d["question"],
                        solution_steps=get_solution_steps_from_response(d["response"]),
                        using_not_trained_model=args.verification_model_name in base_model_names,  # models that are not fine-tuned on our datasets
                    )
                    
                    prompts.append(
                        {"id": d["id"], "prompt": [get_user_message(prompt)]}
                    )
            else:
                raise ValueError(f"Unknown prompt type: {args.verification_prompt_type}")
            
            # get path for saving prompts
            prompt_for_evaluation_path = get_prompt_for_verification_for_sample_and_rank_path(
                dataset_name=args.dataset_name, model_name=args.initial_generation_model_name,
                split="test", prompt_type=args.verification_prompt_type, sample_idx=sample_idx,
            )
        
        # save prompts
        prompt_for_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_for_evaluation_path, "w") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + "\n")
        save_md5_hash(prompt_for_evaluation_path)

        ###
        # verification
        print(f"Running evaluation...")
        evaluation_output_path = get_verification_for_sample_and_rank_outputs_path(
            dataset_name=args.dataset_name,
            initial_response_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            split="test", prompt_type=args.verification_prompt_type, sample_idx=sample_idx
        )
        
        # arguments
        command = [
            f"python src/llm/run_inference.py",
            # load prompts saved above
            f"--dataset_path {prompt_for_evaluation_path}",
            f"--output_path {evaluation_output_path}",
            f"--model_name {args.verification_model_name}",
            f"--batch_size {args.batch_size}",
            f"--max_token {args.max_tokens}",
        ]
        
        if args.not_use_vllm_reward_task:
            command.append("--not_use_vllm_reward_task")
        
        if args.overwrite_cache:
            command.append("--overwrite_cache")
        
        if (args.verification_model_name not in sota_prms_list) \
                and args.not_use_vllm_reward_task:
            # our models
            command.append("--logprobs")
        
        if args.debug:
            command.append("--debug")
        
        # run command
        subprocess.run(" ".join(command), shell=True, text=True)

if __name__ == "__main__":
    main()
