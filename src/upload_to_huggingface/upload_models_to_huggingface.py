from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from src.config import HF_ACCOUNT
from src.upload_to_huggingface.upload_datasets_to_huggingface \
    import UploadToHuggingFaceTap


upload_models_dict = {
    "Llama-3.1-8B-FoVer-PRM": "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_5.0e-6_0430",
    "Qwen-2.5-7B-FoVer-PRM": "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6_0430"
}


model_readme_config_dict = {
    "Llama-3.1-8B-FoVer-PRM": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "license": "llama3.1",
    },
    "Qwen-2.5-7B-FoVer-PRM": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "license": "apache-2.0",
    }
}


def main():
    args = UploadToHuggingFaceTap().parse_args()

    for model_name, model_dir in upload_models_dict.items():
        print(f"Processing {model_name}...")
        repository_name = f"{HF_ACCOUNT}/{model_name}"

        # make repository
        hf_api = HfApi()
        # check if exists
        try:
            hf_api.repo_info(repository_name)
            repo_exists = True
        except Exception as e:
            repo_exists = False
        
        if repo_exists:
            if args.delete_existing_dataset:
                # if exists, delete the model
                print(f"Deleting existing model {repository_name}...")
                hf_api.delete_repo(repository_name, repo_type="model", missing_ok=True)
            else:
                print(f"Model {repository_name} already exists. Skipping...")
                continue
        
        # create the model repository
        print(f"Creating repository {repository_name}...")
        hf_api.create_repo(
            repo_id=repository_name, private=True, repo_type="model"
        )

        # upload readme
        with open("src/upload_to_huggingface/hf_model_readme_metadata.md", "r") as f:
            readme_metadata = f.read().format(
                **model_readme_config_dict[model_name],
            )
        with open("README.md", "r") as f:
            readme_main = f.read()
        readme = readme_metadata + "\n" + readme_main

        hf_api.upload_file(
            repo_type="model", repo_id=repository_name,
            path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")
        )
        
        # upload README images
        hf_api.upload_folder(
            repo_type="model",
            repo_id=repository_name,
            folder_path="readme_figures",
            path_in_repo="readme_figures",
        )
        
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Push the model and tokenizer to the hub
        print(f"Uploading {model_name} to {repository_name}...")
        model.push_to_hub(repository_name, private=True)
        print(f"Uploaded {model_name} to {repository_name}.")
        tokenizer.push_to_hub(repository_name, private=True)
        print(f"Uploaded tokenizer for {model_name} to {repository_name}.")

if __name__ == "__main__":
    main()

