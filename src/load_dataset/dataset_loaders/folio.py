# FOLIO dataset

from datasets import load_dataset, Dataset

from src.downstream_evaluation.prompts.dataset_prompts.folio \
    import get_folio_instruction


def preprocess_folio(example: dict):
    question = get_folio_instruction(
        premise=example["premises"], hypothesis=example["conclusion"]
    )
    example["question"] = question
    
    example_id = example["example_id"]
    example["id"] = f"folio_test_{example_id:05d}"
    
    # True, False, Uncertain
    example["y_true"] = example["label"]
    return example


def get_folio_dataset(split="validation") -> Dataset:
    original_dataset: Dataset = load_dataset("yale-nlp/FOLIO", split=split)
    
    # preprocess
    processed_dataset = original_dataset.map(
        preprocess_folio, remove_columns=original_dataset.column_names
    )
    
    # shuffle instances
    FOLIO = processed_dataset.shuffle(seed=68)
    
    print("FOLIO dataset size:", len(FOLIO))
    
    return FOLIO
