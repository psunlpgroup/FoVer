""" This script generate sample responses for sample-and-rank downstream evaluation on all models on all datasets. """


import subprocess

from tap import Tap

from src.typing import DOWNSTREAM_EVALUATION_MODE
from src.config import base_model_names, downstream_evaluation_datasets_list, \
    downstream_evaluation_for_model_selection_datasets_list


class SampleAndRankTap(Tap):
    evaluation_mode: DOWNSTREAM_EVALUATION_MODE = "final_evaluation"


def main():
    args = SampleAndRankTap().parse_args()
    
    evaluation_datasets_list = {
        "final_evaluation": downstream_evaluation_datasets_list,
        "model_selection": downstream_evaluation_for_model_selection_datasets_list,
    }[args.evaluation_mode]
    
    for verification_prompt_type in ["multi-turn"]:
        for base_model_name in base_model_names:
            for dataset_name in evaluation_datasets_list:
                
                # verification
                verification_arguments_list = [
                    "--base_model_name", base_model_name,
                    "--verification_model_name", "oracle",
                    "--dataset_name", dataset_name,
                    "--verification_prompt_type", verification_prompt_type,
                    "--sample_k", "5",
                ]
                
                subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/get_oracle_scores.py"] + verification_arguments_list)
                subprocess.run(["python", "src/downstream_evaluation/sample_and_rank/get_final_verification_scores.py"] + verification_arguments_list)


if __name__ == "__main__":
    main()
