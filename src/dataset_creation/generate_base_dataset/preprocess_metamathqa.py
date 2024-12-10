from typing import Literal
import json
import re

from tap import Tap
import datasets

from src.config import splits_list
from src.path import get_base_dataset_path
from src.llm.utils import save_md5_hash
from src.dataset_creation.utils.get_dataset_stats import get_basic_dataset_stats


METAMATHQA_BASEDATASET = Literal["gsm8k", "math"]


class PreprocessMetaMathQaTap(Tap):
    base_dataset: METAMATHQA_BASEDATASET = "gsm8k"  # dataset name


dataset_size_dict = {
    "train": 120000,
    "validation": 2000,
    "test": 2000,
}

# we only target questions that are in a similar format to the original datasets
target_augmentation_methods = ["Rephrased", "AnsAug"]


def extract_final_answer_of_metamathqa(text) -> str | None:
    # Match the line that starts with 'The answer is:' and capture the value after it
    match = re.search(r'The answer is:\s*(.*)', text)
    if match:
        return match.group(1).strip()
    return None


def preprocess_answer_of_metamathqa(
        response: str,
        base_dataset: METAMATHQA_BASEDATASET) -> int | str | None:
    
    answer = extract_final_answer_of_metamathqa(response)
    if answer is None:
        return None
    
    if base_dataset == "gsm8k":
        # if not int, return None
        try:
            int(answer)
        except ValueError:
            return None

        return answer
    else:
        # math answer can be in latex format
        return answer


def preprocess_metamathqa_data(
        dataset: list[dict],
        base_dataset: METAMATHQA_BASEDATASET) -> list[dict]:
    
    list_of_dict = []
    for d in dataset:
        processed_answer = preprocess_answer_of_metamathqa(
            d["response"], base_dataset=base_dataset
        )
        
        # we only use simple cases whose answer is int or float
        if processed_answer is None:
            continue
        
        # add to list
        list_of_dict.append(
            {
                "id": d["id"],
                "question": d["query"],
                "y_true": processed_answer,
            }
        )
    
    return list_of_dict    


get_tag_from_base_dataset = {
    "gsm8k": "GSM",
    "math": "MATH",
}


def main():
    args = PreprocessMetaMathQaTap().parse_args()
    
    # big-math only have train split
    # we will split it into train, validation, and test
    raw_data_hf = datasets.load_dataset(
        "meta-math/MetaMathQA", split="train"
    ).shuffle(seed=68)
    
    # save all data with data id
    all_data_path = get_base_dataset_path("metamathqa", split="train")
    if all_data_path.exists():
        with open(all_data_path, "r") as f:
            raw_data = [json.loads(line) for line in f]
    else:
        all_data_path.parent.mkdir(parents=True, exist_ok=True)
        raw_data = []
        with open(all_data_path, "w") as f:
            for idx, d in enumerate(raw_data_hf):
                category = d["type"]
                data_id = f"metamathqa_{category}_{idx:06d}"
                d["id"] = data_id
                
                raw_data.append(d)
                f.write(json.dumps(d) + "\n")
        save_md5_hash(all_data_path)
    
    # filtering
    filtered_dataset: list[dict] = []
    for d in raw_data:
        # filter out examples that are not from the target base dataset
        if not get_tag_from_base_dataset[args.base_dataset] in d["type"]:
            continue
        
        # filter out examples that are not from the target augmentation methods
        if not any(
            method in d["type"]
            for method in target_augmentation_methods
        ):
            continue
        
        filtered_dataset.append(d)
        
        
    
    # convert to list of dict
    # and remove examples whose answer is not int or float
    list_of_dict = preprocess_metamathqa_data(
        filtered_dataset, base_dataset=args.base_dataset
    )

    print(
        f"Number of examples in {args.base_dataset}: {len(list_of_dict)}"
    )

    print(
        f"We will split the dataset into: " \
            f"{dataset_size_dict['train']} train, " \
            f"{dataset_size_dict['validation']} validation, " \
            f"{dataset_size_dict['test']} test"
    )

    # split into train, validation, and test
    if len(list_of_dict) < sum(dataset_size_dict.values()):
        raise ValueError(
            f"Not enough examples in {args.base_dataset}: " \
                f"{len(list_of_dict)} < {sum(dataset_size_dict.values())}"
        )
    
    # split the dataset
    train_data = list_of_dict[:dataset_size_dict["train"]]
    validation_data = list_of_dict[
        -(dataset_size_dict["validation"] + dataset_size_dict["test"]):
        -dataset_size_dict["test"]
    ]
    test_data = list_of_dict[-dataset_size_dict["test"]:]
    
    # save the dataset
    dataset_save_name = f"metamathqa_{args.base_dataset}"
    
    for split, data in zip(
                splits_list, [train_data, validation_data, test_data]
            ):
        
        # save to jsonl
        save_path = get_base_dataset_path(dataset_save_name, split=split)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")
        save_md5_hash(save_path)
    
        # get dataset stats
        stats = get_basic_dataset_stats(data)
        stats_path = save_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
