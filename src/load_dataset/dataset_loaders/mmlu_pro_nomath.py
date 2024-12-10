from datasets import load_dataset, Dataset

from src.downstream_evaluation.prompts.dataset_prompts.mmlu_pro_nomath \
    import get_mmlu_instruction


def preprocess_mmlu_pro_nomath(example: dict):
    question = get_mmlu_instruction(
        question=example["question"], options=example["options"]
    )
    example["question"] = question
    
    example_id = example["question_id"]
    example["id"] = f"mmlu_pro_nomath_test_{example_id:05d}"
    
    # True, False, Uncertain
    example["y_true"] = example["answer"]
    return example


def get_mmlu_pro_nomath_dataset(split="test") -> Dataset:
    if split != "test":
        raise ValueError("we only use test set of mmlu_pro_nomath.")

    original_dataset: Dataset = load_dataset("sam-paech/mmlu-pro-nomath-sml", split=split)
    
    # preprocess
    processed_dataset = original_dataset.map(
        preprocess_mmlu_pro_nomath,
        remove_columns=[c for c in original_dataset.column_names
                        if c != "question"]
    )
    
    # shuffle instances
    MMLU_PRO_NOMATH = processed_dataset.shuffle(seed=68).select(range(250))
    
    print("MMLU_PRO_NOMATH dataset size:", len(MMLU_PRO_NOMATH))
    
    return MMLU_PRO_NOMATH
