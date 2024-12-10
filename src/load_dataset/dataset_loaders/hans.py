# HANS dataset

from pathlib import Path
import random
import json

import pandas as pd
from datasets import Dataset

from src.downstream_evaluation.prompts.dataset_prompts.hans \
    import get_hans_instruction


hans_dataset_dir = Path("../datasets/hans")


def preprocess_hans(example: dict):
    # rename keys
    example["question"] = get_hans_instruction(
        premise=example["sentence1"], hypothesis=example["sentence2"]
    )
    
    example["y_true"] = example["gold_label"]

    example["id"] = f"hans_test_{example['pairID']}"
    
    return example


def get_hans_dataset(split="test") -> Dataset:
    if split != "test":
        raise ValueError("we only use test set of hans.")
    
    ###
    # load dataset from local
    if not hans_dataset_dir.exists():
        raise FileNotFoundError(
            "Dataset directory not found. " \
            f"Clone https://github.com/tommccoy1/HANS to {hans_dataset_dir}"
        )
    
    # load all files
    with open(hans_dataset_dir / "heuristics_evaluation_set.jsonl", "r") as f:
        raw_instances = [json.loads(line) for line in f.readlines()]
    
    # shuffle instances
    raw_instances = random.Random(68).sample(raw_instances, len(raw_instances))
    
    ###
    # postprocess dataset
    raw_dataset = Dataset.from_pandas(pd.DataFrame(raw_instances))
    HANS = raw_dataset.map(preprocess_hans)
    HANS = HANS.select(range(250))
    
    print(f"Split {split} of HANS dataset: {len(HANS)} instances")
    
    return HANS
