""" This code generates outputs from LLMs on the provided dataset """

from typing import Literal
from pathlib import Path

from tqdm import tqdm
import json
import datasets
from torch.utils.data import DataLoader

from src.llm.utils import save_md5_hash
from src.llm.inference_utils import InferenceModel
from src.llm.inference_utils.utils import LlmInferenceParamsTap
from src.llm.inference_utils.call_llm import call_llm


class LlmInferenceTap(LlmInferenceParamsTap):
    dataset_path: str
    output_path: str
    data_format: Literal["json", "csv"] = "json"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: int = 16
    max_tokens: int = 256
    top_p: float = 1
    top_k: int = -1  # -1 means no top_k
    temperature: float = 0.0
    seed: int = 68
    no_vllm: bool = False
    logprobs: bool = False
    overwrite_cache: bool = False
    debug: bool = False

    def process_args(self):
        self.output_dir = Path(self.output_path).parent


if __name__ == "__main__":
    args = LlmInferenceTap().parse_args()

    # Model
    model = InferenceModel(args.model_name, use_vllm=not args.no_vllm, use_vllm_reward_task=not args.not_use_vllm_reward_task)
    
    # Generate responses
    print(f"Generating responses from {args.model_name} for {args.dataset_path}...")

    dataset = datasets.load_dataset(args.data_format, data_files=args.dataset_path, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: list(batch))

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logprobs_file_path = Path(args.output_path).with_suffix(".logprobs.jsonl")

    # Process batches
    responses: list[dict] = []
    with open(args.output_path, "w") as output_file, \
            open(logprobs_file_path, "w") as logprobs_file:
        for batch in tqdm(dataloader):
            batch: list[dict[str, str]] = batch

            # Extract prompts
            prompts_list = [d["prompt"] for d in batch]

            # Generate responses
            responses_list = call_llm(
                model=model,
                prompt=prompts_list,
                overwrite_cache=args.overwrite_cache,
                params=args,
            )

            if not args.logprobs:
                # responses_list is a list of strings
                batch_responses = [
                    {"id": d["id"], "response": response} 
                    for d, response in zip(batch, responses_list)
                ]
                batch_logprobs = []
            else:
                # responses_list is a list of tuples
                batch_responses = [
                    {"id": d["id"], "response": response[0]} 
                    for d, response in zip(batch, responses_list)
                ]
                batch_logprobs = [
                    {"id": d["id"], "logprobs": response[1]} 
                    for d, response in zip(batch, responses_list)
                ]

            # Write batch responses
            for response in batch_responses:
                output_file.write(json.dumps(response) + "\n")
            save_md5_hash(args.output_path)
            
            for logprob in batch_logprobs:
                logprobs_file.write(json.dumps(logprob) + "\n")
            save_md5_hash(logprobs_file_path)

            responses.extend(batch_responses)

            # Debug mode: Stop early
            if len(responses) >= 50 and args.debug:
                break
