""" This script generate sample responses for sample-and-rank downstream evaluation on all models on all datasets. """


import subprocess

from tap import Tap

from src.typing import DOWNSTREAM_EVALUATION_MODE
from src.config import \
    train_dataset_names_list, train_dataset_names_list_multi_turn, \
    base_model_names, downstream_evaluation_datasets_list, \
    downstream_evaluation_for_model_selection_datasets_list, \
    finetuned_verification_models_dict
from src.utils.model_selection import get_best_performance_verifier


class SampleAndRankTap(Tap):
    evaluation_mode: DOWNSTREAM_EVALUATION_MODE = "final_evaluation"
    not_use_vllm_reward_task: bool = False
    overwrite_cache: bool = False


def main():
    args = SampleAndRankTap().parse_args()
    
    evaluation_datasets_list = {
        "final_evaluation": downstream_evaluation_datasets_list,
        "model_selection": downstream_evaluation_for_model_selection_datasets_list,
    }[args.evaluation_mode]

    if args.not_use_vllm_reward_task:
        raise ValueError("This is an old setting.")

    if not args.not_use_vllm_reward_task:
        # this is the current standard setting
        selected_train_dataset_names_list = train_dataset_names_list_multi_turn
    else:
        selected_train_dataset_names_list = train_dataset_names_list

    # generate k responses
    for initial_generation_model in base_model_names:
        for initial_generation_prompt_type in ["few-shot"]:
            
            for dataset_name in evaluation_datasets_list:
                print(f"Generating initial responses for {initial_generation_model} on {dataset_name} with {initial_generation_prompt_type} prompts...")
                
                initial_generation_arguments_list = [
                    "--model_name", initial_generation_model,
                    "--dataset_name", dataset_name,
                    "--prompt_type", initial_generation_prompt_type,
                    "--sample_k", "5",
                    "--temperature", "0.5",  # https://arxiv.org/pdf/2203.11171
                    "--top_k", "40",  # https://arxiv.org/pdf/2203.11171
                ]
                if args.overwrite_cache:
                    initial_generation_arguments_list.append("--overwrite_cache")
                
                # get initial responses
                subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/generate_initial_responses.py"] + initial_generation_arguments_list)
                
                # self-consistency
                print(f"Getting self-consistency answers for {initial_generation_model} on {dataset_name} with {initial_generation_prompt_type} prompts...")
                self_consistency_arguments_list = initial_generation_arguments_list[:-4]
                subprocess.run(["python", "src/downstream_evaluation/self_consistency/get_self_consistency_answers.py"] + self_consistency_arguments_list)

    # verification
    for prm_base_model_name, initial_generation_model in [
            (base_model_name, base_model_name) for base_model_name
            in base_model_names
        ]:

        for initial_generation_prompt_type in ["few-shot"]:
            
            for dataset_name in evaluation_datasets_list:

                # training data for verifiers
                for train_data in [None] + selected_train_dataset_names_list:
                    
                    try:
                        # select evaluation datasets and verification models
                        verification_models_list = []
                        if train_data is None:  # base model
                            verification_models_list.append(prm_base_model_name)
                        else:
                            for optimizer in ["AdamW"]:
                                if args.evaluation_mode == "model_selection":
                                    # list of all candidate verifiers
                                    verification_models_list.extend(
                                        finetuned_verification_models_dict[
                                            prm_base_model_name][optimizer][train_data]
                                    )
                                elif args.evaluation_mode == "final_evaluation":
                                    # get the best verifier
                                    verifier_name = get_best_performance_verifier(
                                        base_model_name=prm_base_model_name,
                                        train_data_name=train_data,
                                        optimizer=optimizer,
                                    )
                                    verification_models_list.append(verifier_name)
                                else:
                                    raise ValueError(f"Invalid evaluation mode: {args.evaluation_mode}")
                    except Exception as e:
                        print(f"Error selecting verification models for {prm_base_model_name} on {train_data}: {e}")
                        continue
                    
                    # verification
                    for verification_model in verification_models_list:
                        if verification_model is None:
                            print(f"No verifier for {train_data} on {prm_base_model_name} found. Skipping...")
                            continue
                        
                        print(f"Running verification for {initial_generation_model} on {dataset_name} with {initial_generation_prompt_type} prompts using {verification_model} verifier...")
                        
                        verification_arguments_list = [
                            "--initial_generation_model_name", initial_generation_model,
                            "--base_model_name", prm_base_model_name,
                            "--verification_model_name", verification_model,
                            "--dataset_name", dataset_name,
                            "--sample_k", "5",
                        ]
                        
                        if not args.not_use_vllm_reward_task:
                            verification_arguments_list.extend(["--verification_prompt_type", "multi-turn"])
                        else:
                            # this is an old setting
                            verification_arguments_list.extend(["--verification_prompt_type", "zero-shot"])
                            verification_arguments_list.append("--not_use_vllm_reward_task")
                        
                        if args.overwrite_cache:
                            verification_arguments_list.append("--overwrite_cache")
                        
                        subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/run_verification.py"] + verification_arguments_list)
                        
                        # postprocess
                        if not args.not_use_vllm_reward_task:
                            subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/postprocess_verification_outputs_sota_prms.py"] + verification_arguments_list)
                        else:
                            subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/postprocess_verification_outputs.py"] + verification_arguments_list)
                        
                        subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/get_final_verification_scores.py"] + verification_arguments_list)


if __name__ == "__main__":
    main()
