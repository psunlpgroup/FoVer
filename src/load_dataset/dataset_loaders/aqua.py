# AQuA-RAT dataset

import functools

from datasets import load_dataset, Dataset

def preprocess_aqua(example: dict, idx: int, split: str):
    example["id"] = f"aqua_{split}_{idx:05d}"
    
    options = " ".join(["(" + o for o in example["options"]])
    example["question"] = "\n".join([example["question"], options])
    example["y_true"] = example["correct"]
    return example


def get_aqua_dataset(split="test") -> Dataset:
    original_dataset: Dataset = load_dataset(
        "deepmind/aqua_rat", split=split
    )
    
    # preprocess
    processed_dataset = original_dataset.map(
        functools.partial(preprocess_aqua, split=split),
        with_indices=True,
    )
    
    # shuffle instances
    AQuA = processed_dataset.shuffle(seed=68)
    
    if split == "train":
        AQuA = AQuA.select(range(10000))
    
    print("AQuA dataset size:", len(AQuA))

    return AQuA
