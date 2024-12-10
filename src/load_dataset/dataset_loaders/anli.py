# FOLIO dataset

from datasets import load_dataset, Dataset

from src.downstream_evaluation.prompts.dataset_prompts.anli \
    import get_anli_instruction


def preprocess_anli(example: dict):
    question = get_anli_instruction(
        premise=example["premise"], hypothesis=example["hypothesis"]
    )
    example["question"] = question
    
    example_id = example["uid"]
    example["id"] = f"anli_test_{example_id}"
    
    # True, False, Uncertain
    example["y_true"] = [
        "neutral", "entailment", "contradiction"][example["label"]]
    return example


def get_anli_dataset(split="test") -> Dataset:
    if split != "test":
        raise ValueError("we only use test set of anli.")

    original_dataset: Dataset = load_dataset("facebook/anli", split="test_r3")
    
    # preprocess
    processed_dataset = original_dataset.map(
        preprocess_anli, remove_columns=original_dataset.column_names
    )
    
    # shuffle instances
    ANLI = processed_dataset.shuffle(seed=68).select(range(250))
    
    print("ANLI dataset size:", len(ANLI))
    
    return ANLI
