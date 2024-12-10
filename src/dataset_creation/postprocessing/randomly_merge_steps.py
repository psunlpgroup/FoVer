""" Randomly merge steps in the dataset to increase the difficulty of the
training data"""

import json
import random

from src.config import splits_list
from src.path import get_error_labels_path
from src.dataset_creation.utils import get_error_labels_stats
from src.dataset_creation.base_dataset_specific.fol.dataset_generation.\
    generate_error_labels import GenerateVerificationDatasetTap
from src.llm.utils import save_md5_hash


class RandomStepMergeTap(GenerateVerificationDatasetTap):
    generation_seed: str = "selected"
    merge_prob: float = 0.2


def merge_solution_steps(step1: str, step2: str,
                         dataset_name: str) -> str:

    if dataset_name == "fldx2_symbol":
        return f"{step1}; {step2}"
    elif dataset_name in ["fldx2_text", "prm800k"]:
        return f"{step1} {step2}"
    elif dataset_name == "isabelle_all":
        return f"{step1}\n{step2}"
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")


def merge_cot_steps(step1: str, step2: str, dataset_name: str) -> str:
    if len(step1) == 0 and len(step2) == 0:
        return ""
    
    # merge
    if dataset_name in ["fldx2_symbol", "fldx2_text", "prm800k", "isabelle_all"]:
        return f"{step1} {step2}"
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")


def main():
    args = RandomStepMergeTap().parse_args()

    for cot_type in ["no_cot", "with_cot"]:
        for split in splits_list[::-1]:
            # load original error labels
            error_labels_path = get_error_labels_path(
                dataset_name=args.dataset_name, model_name=args.model_name,
                split=split, seed=args.generation_seed
            ).with_suffix(f".{cot_type}.jsonl")
            with open(error_labels_path, "r") as f:
                original_error_label_instances = [
                    json.loads(line) for line in f
                ]
            
            ###
            # Randomly merge steps in the dataset to increase the difficulty of
            # the training data
            new_error_label_instances = []
            for d in original_error_label_instances:
                # we will not merge the last step (the final answer)
                # so we need at least 3 steps
                if len(d["proof_steps"]) <= 2:
                    new_error_label_instances.append(d)
                    continue
                
                ###
                # randomly merge steps
                # we only merge at most two steps
                original_proof_steps = d["proof_steps"]
                original_cot_steps = d["cot_steps"]
                original_proof_correctness = d["proof_step_correctness"]
                
                new_proof_steps = []
                new_cot_steps = []
                new_proof_correctness = []
                
                step_idx = 0
                num_merged_steps = 0  # for assertion
                while step_idx < len(original_proof_steps) - 2:
                    random_value = random.Random(
                        f"{args.dataset_name}_{split}_" \
                        f"{args.generation_seed}_{step_idx}"
                    ).random()
                    if random_value < args.merge_prob:
                        num_merged_steps += 1
                        
                        # merge the current step and the next step
                        new_proof_steps.append(
                            merge_solution_steps(
                                original_proof_steps[step_idx],
                                original_proof_steps[step_idx + 1],
                                args.dataset_name
                            )
                        )
                        new_cot_steps.append(
                            merge_cot_steps(
                                original_cot_steps[step_idx],
                                original_cot_steps[step_idx + 1],
                                args.dataset_name
                            )
                        )
                        
                        # the merged step is correct if both steps are correct
                        new_proof_correctness.append(
                            all(
                                [
                                    original_proof_correctness[step_idx],
                                    original_proof_correctness[step_idx + 1]
                                ]
                            )
                        )
                        
                        # skip the next step
                        step_idx += 2
                    else:
                        # keep the current step
                        new_proof_steps.append(original_proof_steps[step_idx])
                        new_cot_steps.append(original_cot_steps[step_idx])
                        new_proof_correctness.append(
                            original_proof_correctness[step_idx]
                        )
                        step_idx += 1
                
                # add the last two steps
                while step_idx < len(original_proof_steps):
                    new_proof_steps.append(original_proof_steps[step_idx])
                    new_cot_steps.append(original_cot_steps[step_idx])
                    new_proof_correctness.append(
                        original_proof_correctness[step_idx]
                    )
                    step_idx += 1
                
                # assertion
                assert len(new_proof_steps) == \
                    len(original_proof_steps) - num_merged_steps, \
                    f"len(new_proof_steps): {len(new_proof_steps)}, " \
                    f"len(original_proof_steps): {len(original_proof_steps)}, " \
                    f"num_merged_steps: {num_merged_steps}"

                # save the new proof steps
                d["proof_steps"] = new_proof_steps
                d["cot_steps"] = new_cot_steps
                d["proof_step_correctness"] = new_proof_correctness
                new_error_label_instances.append(d)
            
            ###
            # save the new error labels
            output_path = error_labels_path.with_suffix(f".step_merged.jsonl")
            with open(output_path, "w") as f:
                for d in new_error_label_instances:
                    f.write(json.dumps(d) + "\n")
            save_md5_hash(output_path)
            
            # get statistics
            stats = get_error_labels_stats(new_error_label_instances)
            
            stats_path = output_path.with_suffix(f".stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
