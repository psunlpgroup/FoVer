""" This script generate sample responses for sample-and-rank downstream evaluation on all models on all datasets. """


import subprocess

from tap import Tap

from src.typing import DOWNSTREAM_EVALUATION_MODE
from src.config import base_model_names, downstream_evaluation_datasets_list, \
    downstream_evaluation_for_model_selection_datasets_list, \
    sota_prms_dict


class SampleAndRankTap(Tap):
    evaluation_mode: DOWNSTREAM_EVALUATION_MODE = "final_evaluation"
    overwrite_cache: bool = False


def main():
    args = SampleAndRankTap().parse_args()
    
    evaluation_datasets_list = {
        "final_evaluation": downstream_evaluation_datasets_list,
        "model_selection": downstream_evaluation_for_model_selection_datasets_list,
    }[args.evaluation_mode]
    
    for base_model_name in base_model_names:
        for initial_generation_prompt_type in ["few-shot"]:
            
            for dataset_name in evaluation_datasets_list:

                # verification
                for verification_model in sota_prms_dict[base_model_name]:

                    # remove this part later
                    if verification_model != "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
                        continue
                    #
                    
                    verification_arguments_list = [
                        "--base_model_name", base_model_name,
                        "--initial_generation_model_name", base_model_name,
                        "--verification_model_name", verification_model,
                        "--dataset_name", dataset_name,
                        "--verification_prompt_type", "multi-turn",
                        "--sample_k", "5",
                    ]
                    
                    if args.overwrite_cache:
                        verification_arguments_list.append("--overwrite_cache")
                    
                    subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/run_verification.py"] + verification_arguments_list)
                    
                    # postprocess
                    subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/postprocess_verification_outputs_sota_prms.py"] + verification_arguments_list)
                    subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/get_final_verification_scores.py"] + verification_arguments_list)


if __name__ == "__main__":
    main()
