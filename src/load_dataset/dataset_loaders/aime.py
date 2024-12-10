# AIME dataset

from datasets import load_dataset, Dataset

from src.downstream_evaluation.evaluation.utils.normalize_answers.\
    gsm8k import string_to_int


def preprocess_aime(example: dict):
    original_id = example["ID"]
    example["id"] = f"aime_{original_id}"
    
    example["question"] = example["Question"]
    example["y_true"] = example["Answer"]

    return example


def check_string_to_int(y_true: str) -> bool:
    """Check if the string can be converted to int."""
    try:
        string_to_int(y_true)
        return True
    except ValueError:
        return False


def get_aime_dataset(split="test") -> Dataset:
    if split != "test":
        raise ValueError("AIME dataset only has test split.")

    original_dataset: Dataset = load_dataset(
        "di-zhang-fdu/AIME_1983_2024", split="train"
    )

    # remove if we get error in string_to_int
    original_dataset = original_dataset.filter(
        lambda x: check_string_to_int(x["Answer"])
    )

    # sort by id (descending)
    original_dataset = original_dataset.sort(
        "ID", reverse=True).select(range(250))
    
    # preprocess
    processed_dataset = original_dataset.map(preprocess_aime)
    
    # shuffle instances
    AIME = processed_dataset.shuffle(seed=68)
    print("AIME dataset size:", len(AIME))

    return AIME
