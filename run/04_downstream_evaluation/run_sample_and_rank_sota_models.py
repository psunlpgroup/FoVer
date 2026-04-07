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
    evaluate_skywork: bool = False
    sample_k: int = 5


def main():
    args = SampleAndRankTap().parse_args()
    
    if args.evaluate_skywork:
        print("Skywork PRM requires running vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B at another terminal. Refer to setup/setup.sh")
    
    evaluation_datasets_list = {
        "final_evaluation": downstream_evaluation_datasets_list,
        "model_selection": downstream_evaluation_for_model_selection_datasets_list,
    }[args.evaluation_mode]
    
    for base_model_name in base_model_names:
        for initial_generation_prompt_type in ["few-shot"]:
            
            for dataset_name in evaluation_datasets_list:
                
                if base_model_name not in sota_prms_dict:
                    print(f"Skipping {base_model_name} as it has no PRM verification models.")
                    continue

                # verification
                for verification_model in sota_prms_dict[base_model_name]:
                    if args.evaluate_skywork:
                        if verification_model != "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
                            continue
                    
                    verification_arguments_list = [
                        "--base_model_name", base_model_name,
                        "--initial_generation_model_name", base_model_name,
                        "--verification_model_name", verification_model,
                        "--dataset_name", dataset_name,
                        "--verification_prompt_type", "multi-turn",
                        "--sample_k", str(args.sample_k),
                    ]
                    
                    if args.overwrite_cache:
                        verification_arguments_list.append("--overwrite_cache")
                    
                    subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/run_verification.py"] + verification_arguments_list)
                    
                    # postprocess
                    subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/postprocess_verification_outputs_multi_turn_and_sota.py"] + verification_arguments_list)
                    subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/get_final_verification_scores.py"] + verification_arguments_list)


if __name__ == "__main__":
    main()
