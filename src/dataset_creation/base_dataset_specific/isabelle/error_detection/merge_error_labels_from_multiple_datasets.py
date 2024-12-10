import json
import random

from tap import Tap

from src.path import get_error_labels_path
from src.config import base_model_names, splits_list
from src.dataset_creation.utils import get_error_labels_stats
from src.llm.utils import save_md5_hash


isabelle_dataset_names_list = [
    "gsm8k", "bigmath_math_word_problems", "metamathqa_gsm8k",
]


def main():
    for base_model_name in base_model_names:
        for split in splits_list:
            merged_error_labels = []
            
            for dataset_name in isabelle_dataset_names_list:
                error_labels_path = get_error_labels_path(
                    dataset_name=f"isabelle_{dataset_name}",
                    model_name=base_model_name,
                    split=split, seed="selected"
                )
                
                if not error_labels_path.exists():
                    print(f"Error labels path does not exist: {error_labels_path}")
                    continue
                
                with open(error_labels_path, "r") as f:
                    error_labels = [json.loads(line) for line in f]
                
                merged_error_labels.extend(error_labels)

            # save merged error labels
            merged_error_labels = random.Random(68).sample(
                merged_error_labels, len(merged_error_labels)
            )
            
            ###
            # import warnings
            # warnings.warn("The merged error labels are sampled to 5x the original size.")
            
            # # copy 5 times
            # merged_error_labels = merged_error_labels * 5
            # # shuffle
            # merged_error_labels = random.Random(68).sample(
            #     merged_error_labels, len(merged_error_labels)
            # )
            # ###
            
            merged_error_labels_path = get_error_labels_path(
                dataset_name=f"isabelle_all",
                model_name=base_model_name,
                split=split, seed="selected"
            )
            
            merged_error_labels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(merged_error_labels_path, "w") as f:
                for error_label in merged_error_labels:
                    f.write(json.dumps(error_label) + "\n")
            save_md5_hash(merged_error_labels_path)
            
            # save stats
            stats_path = merged_error_labels_path.with_suffix(".stats.json")
            stats = get_error_labels_stats(merged_error_labels)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
