import datasets

from src.path import get_base_dataset_path


def get_bigmath_math_word_problems_dataset(split: str) -> datasets.Dataset:
    dataset_path = get_base_dataset_path("bigmath_math_word_problems", split=split)
    dataset = datasets.load_dataset(
        "json", data_files=str(dataset_path)
    )["train"]
    
    return dataset


def get_bigmath_orca_math(split: str) -> datasets.Dataset:
    dataset_path = get_base_dataset_path("bigmath_orca_math_math_word_problems", split=split)
    dataset = datasets.load_dataset(
        "json", data_files=str(dataset_path)
    )["train"]
    
    if split in ["test", "validation"]:
        dataset = dataset.select(range(250))
        
        import warnings
        warnings.warn(
            "Only 250 samples are selected for test and validation."
        )
    
    return dataset
