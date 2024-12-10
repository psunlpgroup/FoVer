import json
from pathlib import Path

import datasets

from src.path import intermediate_dir, get_fover_dataset_path
from src.config import base_model_names, \
    train_dataset_names_list, train_dataset_names_list_multi_turn, \
    splits_list, get_fover_dataset_name
from src.llm.utils import save_md5_hash


llama_factory_datasets_dir = intermediate_dir / "llama_factory_datasets"
dataset_json_path = Path("../LLaMA-Factory-FoVer/data/dataset_info.json")


def convert_to_sharedgpt(dataset: datasets.Dataset) -> list[dict]:
    converted_dataset: list[dict] = []
    
    for example in dataset:
        conversations = []
        
        for message in example["messages"]:
            if message["role"] == "user":
                conversations.append(
                    {"from": "human", "value": message["content"]}
                )
            else:
                conversations.append(
                    {"from": "gpt", "value": message["content"]}
                )
        
        converted_dataset.append({"conversations": conversations})
    
    return converted_dataset


def main():
    for split in splits_list:
        (llama_factory_datasets_dir / split).mkdir(exist_ok=True, parents=True)
    
    with open(dataset_json_path, "r") as f:
        dataset_info = json.load(f)
    
    for model_name in base_model_names:
        for train_dataset_name in train_dataset_names_list_multi_turn:
            print(f"Converting {model_name} on {train_dataset_name}")
            
            if "balanced_last_step" in train_dataset_name:
                # balanced_last_step datasets are only for training
                selected_splits = ["train"]
            else:
                selected_splits = splits_list
            
            # this will be used for dataset name
            hf_dataset_name = get_fover_dataset_name(
                base_model_name=model_name,
                base_dataset_name=train_dataset_name
            )
            
            # we load local dataset
            dataset_dir = get_fover_dataset_path(
                dataset_name=train_dataset_name,
                model_name=model_name, split="train").parent
            
            if not dataset_dir.exists():
                print(f"Dataset directory {dataset_dir} does not exist.")
                continue
            
            try:
                all_dataset = datasets.load_dataset(
                    str(dataset_dir),
                    data_files={
                        split: f"{split}.jsonl" for split in selected_splits
                    }
                )
            except:
                print(f"Failed to load {split} split of {hf_dataset_name}")
                continue
            
            for split in selected_splits:

                dataset = all_dataset[split]
                
                converted_dataset = convert_to_sharedgpt(dataset)
                
                # save converted dataset
                short_hf_dataset_name = hf_dataset_name.split("/")[-1]
                dataset_path = llama_factory_datasets_dir / split \
                    / f"{short_hf_dataset_name}.json"
                with open(dataset_path, "w") as f:
                    json.dump(converted_dataset, f, indent=4)
                save_md5_hash(dataset_path)
                
                # save stats
                stats = {
                    "num_samples": len(converted_dataset),
                }
                stats_path = dataset_path.with_suffix(".stats.json")
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=4)
                
                # update dataset_info.json
                if split == "train":
                    dataset_info[short_hf_dataset_name] = {
                        "file_name": str(Path("../../FoVer" / dataset_path)),
                        "formatting": "sharegpt",
                        "columns": {
                            "messages": "conversations",
                        }
                    }
                elif split == "validation":
                    dataset_info[f"{short_hf_dataset_name}_validation"] = {
                        "file_name": str(Path("../../FoVer" / dataset_path)),
                        "formatting": "sharegpt",
                        "columns": {
                            "messages": "conversations",
                        }
                    }
    
    # save updated dataset_info.json
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == '__main__':
    main()
