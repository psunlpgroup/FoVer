""" Add ground-truth chain-of-thought to error label instances """

import json

from src.config import splits_list
from src.path import get_error_labels_path
from src.llm.utils import save_md5_hash
from src.dataset_creation.base_dataset_specific.fol.dataset_generation.\
    generate_error_labels import GenerateVerificationDatasetTap


class RandomStepMergeTap(GenerateVerificationDatasetTap):
    generation_seed: str = "selected"


# I changed the template to simpler one not to harm out-of-domain tasks
verification_explanation_template = \
    """{proof_steps}"""

def get_verification_cot_from_instance(error_labels_instance: dict) \
        -> tuple[str, list[bool]]:
    """ Get the verification reference for the given error labels instance.
    
    Args:
        error_labels_instance (dict): The error labels instance.
    
    Returns:
        verification_reference (str): The target for training of the
            verification model.
    """
    
    error_labels: list[bool] = error_labels_instance["proof_step_correctness"]

    # explanations
    explanations_list: list[str] = []
    for idx in range(len(error_labels)):
        explanations_list.append(
            verification_explanation_template.format(
                proof_steps=error_labels_instance["proof_steps"][idx],
                # correct_or_incorrect="correct" if error_labels[idx] \
                #     else "incorrect"
            )
        )

    return explanations_list


def main():
    args = RandomStepMergeTap().parse_args()

    for split in splits_list[::-1]:
        # load original error labels
        error_labels_path = get_error_labels_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=split, seed=args.generation_seed
        )
        with open(error_labels_path, "r") as f:
            original_error_label_instances = [
                json.loads(line) for line in f
            ]
        
        ###
        # Train data with no cot
        new_error_label_instances_no_cot = []
        for d in original_error_label_instances:
            d["cot_steps"] = ["" for _ in d["proof_step_correctness"]]
            new_error_label_instances_no_cot.append(d)
        
        # save the new error labels
        output_path_no_cot = error_labels_path.with_suffix(".no_cot.jsonl")
        with open(output_path_no_cot, "w") as f:
            for d in new_error_label_instances_no_cot:
                f.write(json.dumps(d) + "\n")
        save_md5_hash(output_path_no_cot)
        
        ###
        # Add ground-truth chain-of-thought to error label instances
        new_error_label_instances = []
        for d in original_error_label_instances:
            d["cot_steps"] = get_verification_cot_from_instance(d)
            new_error_label_instances.append(d)
        
        # save the new error labels
        output_path = error_labels_path.with_suffix(".with_cot.jsonl")
        with open(output_path, "w") as f:
            for d in new_error_label_instances:
                f.write(json.dumps(d) + "\n")
        save_md5_hash(output_path)


if __name__ == "__main__":
    main()
