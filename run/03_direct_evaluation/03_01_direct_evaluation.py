""" This script runs direct evaluation for all models on all datasets. """


import subprocess

from tap import Tap

from src.config import base_model_names, sota_prms_list, \
    train_dataset_names_list, train_dataset_names_list_multi_turn, \
    get_direct_evaluation_datasets_list
from src.utils.model_selection import get_best_performance_verifier


class DirectEvaluationTap(Tap):
    verification_prompt_type: str = "multi-turn"
    overwrite_cache: bool = False


def main():
    args = DirectEvaluationTap().parse_args()
    
    # evaluation
    for prompt_type in [args.verification_prompt_type]:
        
        selected_train_dataset_names_list = {
            "multi-turn": train_dataset_names_list_multi_turn,
            "zero-shot": train_dataset_names_list,
        }[prompt_type]
        
        for train_data in [None] + selected_train_dataset_names_list:
            for base_model_name in base_model_names + sota_prms_list:
                # sota_prms_list are existing models
                if base_model_name in sota_prms_list:
                    if train_data is not None:
                        continue

                # get list of verification models
                verification_models_list = []
                if train_data is None:  # base model
                    verification_models_list.append(base_model_name)
                else:
                    for optimizer in ["AdamW"]:
                        verifier_name = get_best_performance_verifier(
                            base_model_name=base_model_name,
                            train_data_name=train_data,
                            optimizer=optimizer,
                        )
                        if verifier_name is not None:
                            verification_models_list.append(verifier_name)
                
                for dataset_name in get_direct_evaluation_datasets_list(base_model_name, train_data, verification_prompt_type=prompt_type):
                    for model_name in verification_models_list:
                        print(f"Evaluating {model_name} on {dataset_name} with {prompt_type} prompts...")
                        
                        arguments_list = [
                            "--dataset_name", dataset_name,
                            "--base_model_name", base_model_name,
                            "--verification_model_name", model_name,
                            "--verification_prompt_type", prompt_type
                        ]
                        
                        if args.overwrite_cache:
                            arguments_list.append("--overwrite_cache")
                        
                        if prompt_type != "multi-turn":
                            arguments_list.append("--not_use_vllm_reward_task")
                        
                        # run evaluation
                        subprocess.run(["python", "src/direct_evaluation/run_direct_evaluation.py"] + arguments_list)
                        
                        # postprocess evaluation
                        subprocess.run(["python", "src/direct_evaluation/postprocess_direct_evaluation.py"] + arguments_list)


if __name__ == "__main__":
    main()
