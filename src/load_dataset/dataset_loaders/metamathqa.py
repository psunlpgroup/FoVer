import datasets

from src.path import get_base_dataset_path


def get_metamathqa_dataset(dataset_name: str, split: str) -> datasets.Dataset:
    dataset_path = get_base_dataset_path(dataset_name, split=split)
    dataset = datasets.load_dataset(
        "json", data_files=str(dataset_path)
    )["train"]
    
    # if split in ["test", "validation"]:
    #     dataset = dataset.select(range(250))
        
    #     import warnings
    #     warnings.warn(
    #         "Only 250 samples are selected for test and validation."
    #     )
    
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    return dataset
