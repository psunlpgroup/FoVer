""" Uploads the datasets to Hugging Face Datasets Hub. """

import time

from tap import Tap
import datasets
import datasets.exceptions
from huggingface_hub import HfApi

from src.config import HF_ACCOUNT, splits_list


upload_dataset_dict = {
    "FoVer-FormalLogic-Llama-3.1-8B": "fover_dataset/fldx2_symbol_multi_turn_10k/Llama-3.1-8B-Instruct",
    "FoVer-FormalLogic-Qwen-2.5-7B": "fover_dataset/fldx2_symbol_multi_turn_10k/Qwen2.5-7B-Instruct",
    "FoVer-FormalProof-Llama-3.1-8B": "fover_dataset/isabelle_all_with_cot_10k/Llama-3.1-8B-Instruct",
    "FoVer-FormalProof-Qwen-2.5-7B": "fover_dataset/isabelle_all_with_cot_10k/Qwen2.5-7B-Instruct",
    "FoVer-FormalLogic-FormalProof-Llama-3.1-8B-LastStepBalanced-40k": "fover_dataset/fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k/Llama-3.1-8B-Instruct",
    "FoVer-FormalLogic-FormalProof-Qwen-2.5-7B-LastStepBalanced-40k": "fover_dataset/fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k/Qwen2.5-7B-Instruct",
}


class UploadToHuggingFaceTap(Tap):
    delete_existing_dataset: bool = False


def main():
    args = UploadToHuggingFaceTap().parse_args()
    
    for hf_dataset_name, dataset_dir in upload_dataset_dict.items():
        print(f"Processing {hf_dataset_name}...")
        
        repository_name = f"{HF_ACCOUNT}/{hf_dataset_name}"
        
        hf_api = HfApi()
        # check if exists
        try:
            hf_api.repo_info(repository_name, repo_type="dataset")
            repo_exists = True
        except Exception as e:
            repo_exists = False
        
        if repo_exists:
            if args.delete_existing_dataset:
                # if exists, delete the dataset
                print(f"Deleting existing dataset {repository_name}...")
                hf_api.delete_repo(repository_name, repo_type="dataset", missing_ok=True)
            else:
                print(f"Dataset {repository_name} already exists. Skipping...")
                continue
        
        # create the model repository
        print(f"Creating repository {repository_name}...")
        hf_api.create_repo(
            repo_id=repository_name, private=True, repo_type="dataset"
        )

        # upload readme
        with open("src/upload_to_huggingface/hf_dataset_readme_metadata.md", "r") as f:
            readme_metadata = f.read()
        with open("README.md", "r") as f:
            readme_main = f.read()
        
        if "LastStepBalanced" in hf_dataset_name:
            with open("src/upload_to_huggingface/hf_balanced_dataset_readme.md", "r") as f:
                additional_readme = f.read()
            readme_main = additional_readme + "\n\n" + readme_main

        readme = readme_metadata + "\n" + readme_main

        hf_api.upload_file(
            repo_type="dataset", repo_id=repository_name,
            path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")
        )
        
        # upload README images
        hf_api.upload_folder(
            repo_type="dataset",
            repo_id=repository_name,
            folder_path="readme_figures",
            path_in_repo="readme_figures",
        )
        
        # fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k only has the train split
        selected_train_split = ["train"] if "balanced" in dataset_dir else splits_list

        # for each split, load the dataset and push to hub
        for split in selected_train_split:
            try:
                dataset = datasets.load_dataset(str(dataset_dir), data_files={split: f"{split}.jsonl"}, split=split)
            except FileNotFoundError:
                print(f"Dataset {dataset_dir} for split {split} not found.")
                continue
            except datasets.exceptions.DatasetGenerationError as e:
                print(f"Error loading dataset {dataset_dir} for split {split}: {e}")
                continue
            
            # push the dataset
            print(f"Pushing dataset {repository_name}...")
            dataset.push_to_hub(repository_name, split=split, private=True)
        
        # pose 30 seconds to avoid rate limit
        print(f"Sleeping for 30 seconds to avoid rate limit...")
        time.sleep(30)


if __name__ == "__main__":
    main()
