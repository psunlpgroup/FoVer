# MATH dataset

from pathlib import Path
import random
import re
import json

import pandas as pd
from datasets import Dataset

from src.downstream_evaluation.evaluation.utils.extract_final_answer \
    import extract_from_box


math_dataset_dir = Path("../datasets/MATH")


def preprocess_math(example: dict):
    # rename keys
    example["question"] = example.pop("problem")
    
    # extract final answer
    solution: str = example["solution"]
    # get \\boxed{y_true}
    
    # math can include sentences after the final answer
    extracted_string = None
    for solution_sentence in solution.split(".\n"):
        extracted_string = extract_from_box(solution_sentence)
        if extracted_string is not None:
            break
    
    example["y_true"] = extracted_string
    
    return example


def get_math_dataset(split="test") -> Dataset:
    # MATH dataset does not have validation split
    load_split = "train" if split == "train" else "test"
    
    ###
    # load dataset from local
    math_dataset_split_dir = math_dataset_dir / split
    if not math_dataset_split_dir.exists():
        raise FileNotFoundError(
            "Dataset directory not found. " \
            f"Download MATH dataset to {math_dataset_dir}"
        )
    
    # each instance is in a separate json file
    # math dataset has subdirectories for different types of problems
    subdirectories = [x for x in math_dataset_split_dir.iterdir() if x.is_dir()]
    all_files: list[Path] = []
    for subdirectory in subdirectories:
        all_files.extend(
            [file_path for file_path in subdirectory.iterdir()
             if file_path.is_file() and file_path.suffix == '.json']
        )
    
    # load all files
    raw_instances = []
    for file in all_files:
        with open(file, "r") as f:
            instance = json.load(f)
        question_type: str = instance["type"]
        data_id = f"math_{question_type.lower()}_{file.stem}"
        
        instance["id"] = data_id
        raw_instances.append(instance)
    
    # make sure that the order is consistent
    raw_instances = sorted(raw_instances, key=lambda x: x["id"])
    
    # shuffle instances
    raw_instances = random.Random(68).sample(raw_instances, len(raw_instances))
    
    ###
    # postprocess dataset
    raw_dataset = Dataset.from_pandas(pd.DataFrame(raw_instances))
    MATH = raw_dataset.map(preprocess_math)
    
    if split == "test":
        MATH = MATH.select(range(250))
    elif split == "validation":
        MATH = MATH.select(range(500, 750))
    
    print(f"Split {split} of MATH dataset: {len(MATH)} instances")
    
    return MATH
