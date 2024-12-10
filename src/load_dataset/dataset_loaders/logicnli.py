# logicnli dataset

import functools

from datasets import load_dataset, Dataset

from src.downstream_evaluation.prompts.dataset_prompts.logicnli \
    import get_logicnli_instruction


def preprocess_logicnli(example: dict, idx: int, split: str):
    question = get_logicnli_instruction(
        premise=example["premise"], hypothesis=example["hypothesis"]
    )
    example["question"] = question
    
    example["id"] = f"logicnli_{split}_{idx:05d}"
    
    # True, False, Uncertain
    example["y_true"] = example["label"]
    return example


def get_logicnli_dataset(split="validation") -> Dataset:
    original_dataset: Dataset = load_dataset("tasksource/LogicNLI", split=split)
    
    # preprocess
    processed_dataset = original_dataset.map(
        functools.partial(preprocess_logicnli, split=split),
        remove_columns=original_dataset.column_names,
        with_indices=True
    )

    # filter out
    # we only use labels entailment, contradiction, and neutral
    processed_dataset = processed_dataset.filter(
        lambda x: x["y_true"] in ["entailment", "contradiction", "neutral"]
    )
    
    # shuffle instances
    logicnli = processed_dataset.shuffle(seed=68)
    
    if split in ["test", "validation"]:
        logicnli = logicnli.select(range(250))
    
    print("logicnli dataset size:", len(logicnli))
    
    return logicnli
