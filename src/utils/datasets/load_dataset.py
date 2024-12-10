import datasets
from huggingface_hub import list_datasets


def load_dataset(dataset_name: str, split: str) -> datasets.Dataset:
    """ Load the dataset with the given name and split.
    For local datasets, the dataset is loaded from the local jsonl file.
    Update this function if you want to add other options for local datasets.
    
    This function is needed because our local datasets
    (e.g., direct_evaluation_datasets) includes test.jsonl.md5 files, which
    causes an error in datasets.load_dataset.
    """
    
    if len(list(list_datasets(search=dataset_name))) > 0:
        # if available in huggingface hub
        return datasets.load_dataset(dataset_name, split=split)
    else:
        # load local dataset
        return datasets.load_dataset(
            "json", data_files=f"{dataset_name}/{split}.jsonl",
            split="train"  # this is a psudo-split
        )
