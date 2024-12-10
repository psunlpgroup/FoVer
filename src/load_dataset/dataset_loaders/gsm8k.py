# GSM8k dataset

from functools import partial

from datasets import load_dataset, Dataset

def preprocess_gsm8k(example: dict, idx: int, split: str):
    example["id"] = f"gsm8k_{split}_{idx:05d}"
    original_answer: str = example["answer"]
    example["y_true"] = original_answer.split("#### ")[-1]
    return example


def get_gsm8k_dataset(split="test") -> Dataset:
    # gsm8k does not have a validation split
    load_split = "train" if split == "train" else "test"
    
    original_dataset: Dataset = load_dataset("openai/gsm8k", "main",
                                             split=load_split)
    
    # preprocess
    processed_dataset = original_dataset.map(
        partial(preprocess_gsm8k, split=split),
        with_indices=True
    )
    
    # shuffle instances
    GSM8K = processed_dataset.shuffle(seed=68)
    
    print("GSM8K dataset size:", len(GSM8K))

    if split == "test":
        GSM8K = GSM8K.select(range(250))
    if split == "validation":
        # use a part of the test set as validation set
        GSM8K = GSM8K.select(range(500, 750))

    print(f"Split {split} of GSM8K dataset size:", len(GSM8K))
    
    return GSM8K
