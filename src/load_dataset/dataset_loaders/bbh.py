# GSM8k dataset

from pathlib import Path
import json

import pandas as pd
from datasets import Dataset


bbh_dataset_dir = Path("../datasets/BIG-Bench-Hard/bbh")


def preprocess_bbh(example: dict, idx: int, subset: str):
    example["id"] = f"bbh_{subset}_{idx:05d}"
    example["question"] = example["input"]
    example["y_true"] = example["target"]
    return example


def get_bbh_dataset(subset: str) -> Dataset:
    ###
    # load dataset from local
    bbh_dataset_split_dir = bbh_dataset_dir / f"{subset}.json"
    with open(bbh_dataset_split_dir, "r") as f:
        raw_instances = json.load(f)["examples"]
    
    # preprocess
    raw_dataset = Dataset.from_pandas(pd.DataFrame(raw_instances))
    BBH = raw_dataset.map(
        preprocess_bbh, with_indices=True, fn_kwargs={"subset": subset}
    )
    
    print(f"BBH {subset} dataset size:", len(BBH))
    
    return BBH
