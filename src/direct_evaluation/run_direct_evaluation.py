""" This code generates initial answers from LLMs on base datasets. We will automatically verify the outputs to create the datasets with error labels. """

from typing import Literal
import subprocess
import json

from tap import Tap

from src.path import get_prompt_for_direct_evaluation_path, get_direct_evaluation_outputs_path
from src.config import sota_prms_list
from src.typing import BASE_MODEL
from src.utils.datasets import load_dataset
from src.utils.sota_prms import get_verification_prompt_for_sota_prms
# from src.dataset_creation.initial_answer_generation.prompts import get_evaluation_few_shot_prompt
from src.llm.utils import save_md5_hash


class DirectEvaluationTap(Tap):
    dataset_name: str  # name or path of the dataset
    base_model_name: BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    verification_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # path to fine-tuned model
    batch_size: int = 16
    max_tokens: int = 2048
    max_num_evaluation_instances: int | None = None
    verification_prompt_type: Literal["zero-shot", "few-shot", "multi-turn"] = "multi-turn"
    not_use_vllm_reward_task: bool = False  # use vllm "reward" task for generation
    overwrite_cache: bool = False  # do not use cache
    debug: bool = False


def main():
    args = DirectEvaluationTap().parse_args()
    print(args)
    
    if args.verification_prompt_type == "multi-turn" and args.not_use_vllm_reward_task:
        raise ValueError("Multi-turn prompt type should be used without --not_use_vllm_reward_task.")
    
    splits_list = ["test"]
    if "fover" in args.dataset_name:  # our dataset
        splits_list.append("train")
    
    for split in splits_list:
        ###
        # create and save prompts
        dataset = load_dataset(args.dataset_name, split=split)
        if args.max_num_evaluation_instances is not None:
            if len(dataset) > args.max_num_evaluation_instances:
                dataset = dataset.shuffle(seed=68)
                dataset = dataset.select(range(args.max_num_evaluation_instances))
        print(f"Loaded {len(dataset)} instances from {args.dataset_name} dataset.")
        
        prompt_for_evaluation_path = get_prompt_for_direct_evaluation_path(
            dataset_name=args.dataset_name, model_name=args.base_model_name, split=split, prompt_type=args.verification_prompt_type
        )
        prompt_for_evaluation_path.parent.mkdir(parents=True, exist_ok=True)

        # generate prompts
        print(f"Generating prompts for evaluation...")
        prompts: list[dict] = []
        for d in dataset:
            if args.verification_prompt_type == "few-shot":
                raise NotImplementedError("Few-shot prompts are not implemented yet.")
            elif args.verification_prompt_type == "zero-shot":
                if d["messages"][-1]["role"] != "user":
                    # remove assistant's last message, which is the target
                    prompt = d["messages"][:-1]
                else:
                    # only inlucing user's messages
                    prompt = d["messages"]
            elif args.verification_prompt_type == "multi-turn":
                if args.verification_model_name in sota_prms_list:
                    # existing sota models
                    prompt = get_verification_prompt_for_sota_prms(
                        problem=d["problem"],
                        solution_steps=d["solution_steps"],
                        model_name=args.verification_model_name,
                    )
                else:
                    from src.prm.preprocessing import get_fover_input_format
                    prompt = get_fover_input_format(
                        problem=d["problem"],
                        solution_steps=d["solution_steps"],
                        reference_error_labels=None,  # prediction input
                        model_role_name="assistant" if "gemma" not in args.base_model_name else "gemma",
                    )
            else:
                raise ValueError(f"Unknown prompt type: {args.verification_prompt_type}")
            
            prompts.append({"id": d["id"], "prompt": prompt})
        
        # save prompts
        with open(prompt_for_evaluation_path, "w") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + "\n")
        save_md5_hash(prompt_for_evaluation_path)

        ###
        # evaluation
        print(f"Running evaluation...")
        evaluation_output_path = get_direct_evaluation_outputs_path(
            dataset_name=args.dataset_name,
            model_name=args.verification_model_name,
            split=split, prompt_type=args.verification_prompt_type
        )
        command = [
            f"python src/llm/run_inference.py",
            f"--dataset_path {prompt_for_evaluation_path}",  # load prompts saved above
            f"--output_path {evaluation_output_path}",
            f"--model_name {args.verification_model_name}",
            f"--batch_size {args.batch_size}",
            f"--max_token {args.max_tokens}",
            f"--overwrite_cache" if args.overwrite_cache else "",
        ]
        
        if args.verification_prompt_type == "zero-shot":
            command.append("--logprobs")
            command.append("--not_use_vllm_reward_task")
        
        if args.debug:
            command.append("--debug")
        
        subprocess.run(" ".join(command), shell=True, text=True)

if __name__ == "__main__":
    main()
