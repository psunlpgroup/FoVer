import json
from pathlib import Path
import subprocess
import shutil

from tap import Tap

from src.config import splits_list
from src.typing import BASE_MODEL
from src.path import get_initial_answers_path, model_inputs_dir, \
    model_responses_dir
from src.dataset_creation.base_dataset_specific.isabelle.\
    informal_to_formal.get_few_shot_prompt import \
        get_few_shot_prompt_for_statement_conversion, \
        get_few_shot_prompt_for_proof_conversion
from src.llm.utils import save_md5_hash


class IsabelleInformalToFormalTap(Tap):
    dataset_name: str = "gsm8k"
    base_model_name: BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    conversion_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens: int = 2048  # maximum number of tokens to generate
    batch_size: int = 16  # batch size for generation
    overwrite_cache: bool = False


def get_prompt_for_informal_to_formal_conversion_path(
        dataset_name: str,
        initial_generation_model_name: str, conversion_model: str,
        split: str, statement_or_proof: str,
        seed: str = "selected") -> Path:
    """ Get the path to the JSONL file of the prompts for initial generation
    for the given dataset, model, and split. """
    
    initial_short_name = initial_generation_model_name.split("/")[-1]
    conversion_short_name = conversion_model.split("/")[-1]
    return model_inputs_dir / "isabelle" / "informal_to_formal_conversion" / \
        statement_or_proof / dataset_name / \
        f"initial_generation={initial_short_name}" / \
        f"conversion={conversion_short_name}" / \
        f"seed={seed}" / f"{split}.jsonl"


def get_converted_formal_statement_or_proofs_path(
        dataset_name: str,
        initial_generation_model_name: str, conversion_model_name: str,
        statement_or_proof: str,
        split: str) -> Path:
    """ Get the path to the JSONL file of the converted formal proofs for the
    given dataset, model, and split. """
    
    initial_short_name = initial_generation_model_name.split("/")[-1]
    conversion_short_name = conversion_model_name.split("/")[-1]
    return model_responses_dir / "isabelle" / \
        "informal_to_formal_conversion" / statement_or_proof / dataset_name / \
        f"initial_generation={initial_short_name}" / \
        f"conversion={conversion_short_name}" / f"{split}.jsonl"


def get_converted_formal_proofs_thy_file_path(
        dataset_name: str,
        initial_generation_model_name: str, conversion_model_name: str,
        split: str, data_id: str) -> Path:
    """ Get the path to the thy file of the converted formal proofs for the
    given dataset, model, and split. """
    
    initial_short_name = initial_generation_model_name.split("/")[-1]
    conversion_short_name = conversion_model_name.split("/")[-1]
    return model_responses_dir / "isabelle" / "formal_proofs" / \
        dataset_name / f"initial_generation={initial_short_name}" / \
        f"conversion={conversion_short_name}" / split / f"{data_id}.thy"


def main():
    args = IsabelleInformalToFormalTap().parse_args()
    
    for split in splits_list:
        # We convert informal proofs to formal proofs (isabelle) in two steps:
        # 1. Generate formal statement
        # 2. Convert the formal statement to a formal proof
        
        # This is the process proposed in the dtv paper.
        
        # load informal data
        initial_answers_path = get_initial_answers_path(
            dataset_name=args.dataset_name, model_name=args.base_model_name,
            split=split, seed="selected"
        )
        with open(initial_answers_path, "r") as f:
            informal_data = [json.loads(line) for line in f]
        
        # ###
        # # debug
        # informal_data = informal_data[:16]
        # ###

        ###
        # save prompts for statement generation
        # create prompt
        statement_prompts_list = []
        for d in informal_data:
            prompt = get_few_shot_prompt_for_statement_conversion(
                dataset_name=args.dataset_name,
                model_name=args.conversion_model_name,
                informal_data=d
            )
            statement_prompts_list.append(
                {
                    "id": d["id"],
                    "prompt": prompt,
                }
            )
        
        # save prompts
        statement_prompt_path = \
            get_prompt_for_informal_to_formal_conversion_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model=args.conversion_model_name,
                split=split,
                statement_or_proof="statement"
            )
        statement_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(statement_prompt_path, "w") as f:
            for prompt in statement_prompts_list:
                f.write(json.dumps(prompt) + "\n")
        save_md5_hash(statement_prompt_path)
        
        ###
        # run inference
        formal_statement_output_path = get_converted_formal_statement_or_proofs_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
            statement_or_proof="statement",
            split=split
        )
        formal_statement_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        arguments_list = [
            "--dataset_path", statement_prompt_path,
            "--output_path", formal_statement_output_path,
            "--model_name", args.conversion_model_name,
            "--batch_size", str(args.batch_size),
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(0.0),
        ]
        if args.overwrite_cache:
            arguments_list.append("--overwrite_cache")
        subprocess.run(["python", "src/llm/run_inference.py"] + arguments_list)

        ###
        # save prompts for proof generation
        
        # get formal statements from the first step
        with open(formal_statement_output_path, "r") as f:
            formal_statement_responses = [json.loads(line) for line in f]
        formal_statements = [d["response"] for d in formal_statement_responses]
        
        # create prompt
        proof_prompts_list = []
        for idx, d in enumerate(informal_data):
            formal_statement = formal_statements[idx]
            
            prompt = get_few_shot_prompt_for_proof_conversion(
                dataset_name=args.dataset_name,
                model_name=args.conversion_model_name,
                informal_data=d,
                formal_statement=formal_statement
            )
            proof_prompts_list.append(
                {
                    "id": d["id"],
                    "prompt": prompt,
                }
            )
        
        # save prompts
        proof_prompt_path = \
            get_prompt_for_informal_to_formal_conversion_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model=args.conversion_model_name,
                statement_or_proof="proof",
                split=split
            )
        proof_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proof_prompt_path, "w") as f:
            for prompt in proof_prompts_list:
                f.write(json.dumps(prompt) + "\n")
        save_md5_hash(proof_prompt_path)

        ###
        # run inference
        formal_proof_output_path = \
            get_converted_formal_statement_or_proofs_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                statement_or_proof="proof",
                split=split
            )
        formal_proof_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        arguments_list = [
            "--dataset_path", proof_prompt_path,
            "--output_path", formal_proof_output_path,
            "--model_name", args.conversion_model_name,
            "--batch_size", str(args.batch_size),
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(0.0),
        ]
        if args.overwrite_cache:
            arguments_list.append("--overwrite_cache")
        subprocess.run(["python", "src/llm/run_inference.py"] + arguments_list)
        
        # save as thy files
        with open(formal_proof_output_path, "r") as f:
            formal_data = [json.loads(line) for line in f]
        
        for idx, d in enumerate(formal_data):
            data_id = d["id"]
            formal_proof = d["response"]
            formal_proof_thy_path = get_converted_formal_proofs_thy_file_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                split=split,
                data_id=data_id
            )
            
            # remove the existing directory
            if idx == 0:
                if formal_proof_thy_path.parent.exists():
                    # remove the existing directory
                    shutil.rmtree(formal_proof_thy_path.parent)
                formal_proof_thy_path.parent.mkdir(parents=True, exist_ok=True)
            
            # save the formal proof as a thy file
            with open(formal_proof_thy_path, "w") as f:
                f.write(formal_proof)


if __name__ == "__main__":
    main()
