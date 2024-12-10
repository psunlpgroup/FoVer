""" Generate initial k responses for the sample-and-rank. """

import subprocess
import json

from tap import Tap

from src.path import get_prompt_for_downstream_evaluation_initial_responses_path, \
    get_downstream_evaluation_initial_responses_path, \
    get_prompt_for_extracting_answers_from_downstream_evaluation_initial_responses_path
from src.llm.utils import save_md5_hash
from src.load_dataset import load_existing_dataset
from src.downstream_evaluation.prompts import \
    get_fewshot_prompt_for_initial_generation_of_sample_and_rank, \
    get_fewshot_prompt_for_answer_extraction_of_sample_and_rank


class SampleAndRankPromptGenerationTap(Tap):
    dataset_name: str  # dataset name
    model_name: str  # model name
    prompt_type: str = "few-shot"  # prompt type (only few-shot is supported)


class SampleAndRankInitialResponsesTap(SampleAndRankPromptGenerationTap):
    sample_k: int  # number of responses to generate
    temperature: float  # temperature for sampling
    top_k: int  # top k for sampling
    max_tokens: int = 2048  # maximum number of tokens to generate
    batch_size: int = 16  # batch size for generation
    answer_extraction_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    overwrite_cache: bool = False
    split: str = "test"


def main():
    args = SampleAndRankInitialResponsesTap().parse_args()

    print(args)

    ###
    # generate prompts
    if args.prompt_type == "few-shot":
        # get few-shot examples in chat format (list of dictionaries)
        get_prompt = get_fewshot_prompt_for_initial_generation_of_sample_and_rank
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")

    # load dataset
    print(f"Loading {args.split} split of {args.dataset_name} dataset...")
    dataset = load_existing_dataset(args.dataset_name, split=args.split)
    print(dataset[0])
    print(f"Dataset size: {len(dataset)}")
    
    # create prompts
    print("Creating prompts...")
    prompts = []
    for d in dataset:
        # add new input to the few-shot examples
        prompt = get_prompt(
            dataset_name=args.dataset_name, model_name=args.model_name,
            new_question=d["question"]
        )
        
        prompts.append({"id": d["id"], "prompt": prompt})
    
    # save prompts
    prompt_path = get_prompt_for_downstream_evaluation_initial_responses_path(
        dataset_name=args.dataset_name, model_name=args.model_name,
        split=args.split, prompt_type=args.prompt_type
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
    save_md5_hash(prompt_path)

    ###
    # generate responses
    print("Generating responses...")
    for sample_idx in range(args.sample_k):
        print(f"Sample {sample_idx + 1}/{args.sample_k}")
        output_path = get_downstream_evaluation_initial_responses_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=args.split, prompt_type=args.prompt_type, sample_idx=sample_idx
        )
        
        arguments_list = [
            "--dataset_path", prompt_path,
            "--output_path", output_path,
            "--model_name", args.model_name,
            "--batch_size", str(args.batch_size),
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--top_k", str(args.top_k),
            "--seed", str(sample_idx),
        ]
        if args.overwrite_cache:
            arguments_list.append("--overwrite_cache")
        subprocess.run(["python", "src/llm/run_inference.py"] + arguments_list)

        ###
        # extract answers (postprocessing)
        
        # load model responses (solutions)
        with open(output_path, "r") as f:
            responses = [json.loads(line) for line in f]
        
        # generate prompts
        print("Creating prompts for answer extraction...")
        answer_extraction_prompts = []
        for r in responses:
            prompt = get_fewshot_prompt_for_answer_extraction_of_sample_and_rank(
                dataset_name=args.dataset_name, model_name=args.answer_extraction_model,
                new_solution=r["response"]
            )
            
            answer_extraction_prompts.append({"id": r["id"], "prompt": prompt})
        
        # save prompts
        answer_extraction_prompt_path = get_prompt_for_extracting_answers_from_downstream_evaluation_initial_responses_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=args.split, sample_idx=sample_idx
        )
        answer_extraction_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(answer_extraction_prompt_path, "w") as f:
            for prompt in answer_extraction_prompts:
                f.write(json.dumps(prompt) + "\n")
        save_md5_hash(answer_extraction_prompt_path)
        
        # run answer extraction
        arguments_list = [
            "--dataset_path", answer_extraction_prompt_path,
            "--output_path", output_path.with_suffix(".postprocessed.jsonl"),
            "--model_name", args.answer_extraction_model,
            "--batch_size", str(args.batch_size),
            "--max_tokens", str(128),
            "--temperature", str(0.0),
        ]
        if args.overwrite_cache:
            arguments_list.append("--overwrite_cache")
        subprocess.run(["python", "src/llm/run_inference.py"] + arguments_list)


if __name__ == "__main__":
    main()
