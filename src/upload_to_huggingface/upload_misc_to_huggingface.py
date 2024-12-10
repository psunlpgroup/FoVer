""" Uploads the datasets to Hugging Face Datasets Hub. """

import time

from huggingface_hub import HfApi

from src.config import HF_ACCOUNT


upload_directories_list = [
    "fover_dataset",
    "intermediate_outputs",
    "model_inputs",
    # "model_responses",
]


def main():
    repository_name = f"{HF_ACCOUNT}/FoVer-misc"
    
    hf_api = HfApi()
    # check if exists
    try:
        hf_api.repo_info(repository_name, repo_type="dataset")
        repo_exists = True
    except Exception as e:
        repo_exists = False
    
    if repo_exists:
        raise Exception(
            f"Repository {repository_name} already exists. "\
            "Please delete it manually if you want to re-upload the materials."
        )
            
    # create repository
    print(f"Creating repository {repository_name}...")
    hf_api.create_repo(
        repo_id=repository_name, private=True, repo_type="dataset"
    )

    # upload readme
    with open("src/upload_to_huggingface/hf_misc_readme.md", "r") as f:
        readme_metadata = f.read()
    with open("README.md", "r") as f:
        readme_main = f.read()
    readme = readme_metadata + "\n\n" + readme_main

    hf_api.upload_file(
        repo_type="dataset", repo_id=repository_name,
        path_in_repo="README.md",
        path_or_fileobj=readme.encode("utf-8")
    )
    
    # upload README images
    hf_api.upload_folder(
        repo_type="dataset",
        repo_id=repository_name,
        folder_path="readme_figures",
        path_in_repo="readme_figures",
    )

    # upload directories
    for directory_name in upload_directories_list:
        print(f"Processing {directory_name}...")

        # upload
        hf_api.upload_folder(
            repo_type="dataset",
            repo_id=repository_name,
            folder_path=directory_name,
            path_in_repo=directory_name,
        )
        
        # pose 30 seconds to avoid rate limit
        print(f"Sleeping for 30 seconds to avoid rate limit...")
        time.sleep(30)


if __name__ == "__main__":
    main()
