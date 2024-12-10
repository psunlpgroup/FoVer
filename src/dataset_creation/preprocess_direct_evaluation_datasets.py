""" Preprocess direct evaluation datasets (e.g., ProcessBench, PRM800) """

import json

import datasets

from src.config import processbench_splits
from src.path import get_direct_evaluation_dataset_path
from src.dataset_creation.prompts import get_user_message, get_verification_prompt_for_single_turn_data
from src.llm.utils import save_md5_hash
from src.dataset_creation.utils import get_prm_dataset_stats


def get_prompt_with_chat_template(prompt_text: str) -> list[dict[str, str]]:
    return [get_user_message(prompt_text)]


def main():
    for split in processbench_splits:
        print(f"Preprocessing ProcessBench {split}...")
        
        dataset = datasets.load_dataset("Qwen/ProcessBench", split=split)
        dataset_name = f"processbench_{split}"
        
        # preprocess dataset
        evaluation_dataset: list[dict] = []
        for idx, instance in enumerate(dataset):
            instance_id = f"{dataset_name}_{idx}"
            
            # old dataset format
            # prompt text
            prompt_text = get_verification_prompt_for_single_turn_data(
                data_id=instance_id, problem=instance["problem"], solution_steps=instance["steps"]
            )
            messages = get_prompt_with_chat_template(prompt_text)
            
            # error labels
            first_error_step = int(instance["label"])
            error_labels = []
            for i in range(len(instance["steps"])):
                if first_error_step == -1:
                    # all steps are correct
                    error_labels.append(True)
                else:
                    # otherwise, the first error step is annotated
                    if i < first_error_step:
                        error_labels.append(True)
                    elif i == first_error_step:
                        # the first error step
                        error_labels.append(False)
                    else:
                        # subsecuent steps are not annotated
                        error_labels.append(None)
            
            # add instance to evaluation dataset
            evaluation_dataset.append(
                {
                    "id": instance_id,
                    "messages": messages,
                    "error_labels": error_labels,
                    "base_dataset": dataset_name,
                    "problem": instance["problem"],
                    "solution_steps": instance["steps"],
                }
            )
        
        # save evaluation dataset
        evaluation_dataset_path = get_direct_evaluation_dataset_path(dataset_name, "test")
        evaluation_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(evaluation_dataset_path, "w") as f:
            for instance in evaluation_dataset:
                f.write(json.dumps(instance) + "\n")
        save_md5_hash(evaluation_dataset_path)
        
        # get stats
        stats = get_prm_dataset_stats(evaluation_dataset)
        stats_path = evaluation_dataset_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
