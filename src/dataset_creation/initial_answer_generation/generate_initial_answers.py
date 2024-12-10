""" This code generates initial answers from LLMs on base datasets. We will automatically verify the outputs to create the datasets with error labels. """

import subprocess
import json

from tap import Tap
import datasets

from src.typing import SPLIT, BASE_MODEL
from src.path import get_base_dataset_path, get_prompt_for_initial_generation_path, get_initial_answers_path
from src.dataset_creation.initial_answer_generation.prompts import get_initial_generation_prompt
from src.llm.utils import save_md5_hash


class DatasetCreationTap(Tap):
    dataset_name: str = "fldx2_symbol"
    num_samples: int = 3  # number of samples to generate for each instance


class GenerateInitialAnswersTap(DatasetCreationTap):
    model_name: BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    split: SPLIT
    batch_size: int = 16
    max_tokens: int = 256
    overwrite_cache: bool = False  # do not use cache
    debug: bool = False


if __name__ == "__main__":
    args = GenerateInitialAnswersTap().parse_args()
    
    base_dataset_path = get_base_dataset_path(dataset_name=args.dataset_name, split=args.split)
    dataset = datasets.load_dataset("json", data_files=str(base_dataset_path))["train"]  # "train" is a dummy split name. the actual split is specified in the path.
    
    # we generate multiple samples for each instance to select challenging
    # error cases for training the verifier
    for seed in range(1, args.num_samples + 1):
        print(f"Generating initial answers for seed {seed} / {args.num_samples}")
        
        ###
        # create and save prompts
        prompt_for_initial_generation_path = get_prompt_for_initial_generation_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=args.split,
            seed=seed
        )
        prompt_for_initial_generation_path.parent.mkdir(parents=True, exist_ok=True)

        # generate prompts
        prompts: list[dict] = []
        for d in dataset:
            prompt = get_initial_generation_prompt(
                dataset_name=args.dataset_name, model_name=args.model_name,
                instance=d,
                seed=seed  # select different few-shot examples for each seed
            )
            prompts.append({"id": d["id"], "prompt": prompt})
        
        # save prompts
        with open(prompt_for_initial_generation_path, "w") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + "\n")
        save_md5_hash(prompt_for_initial_generation_path)

        ###
        # generate initial answers
        initial_answer_path = get_initial_answers_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=args.split, seed=seed
        )
        command = [
            f"python src/llm/run_inference.py",
            f"--dataset_path {prompt_for_initial_generation_path}",  # load prompts saved above
            f"--output_path {initial_answer_path}",
            f"--model_name {args.model_name}",
            f"--batch_size {args.batch_size}",
            f"--max_token {args.max_tokens}",
            f"--overwrite_cache" if args.overwrite_cache else "",
        ]
        
        if args.debug:
            command.append("--debug")
    
        subprocess.run(" ".join(command), shell=True, text=True)
