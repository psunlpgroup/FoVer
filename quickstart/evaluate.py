from typing import Literal
from pathlib import Path
import subprocess
import shutil

from tap import Tap

from src.typing import BASE_MODEL
from src.path import get_direct_evaluation_metrics_path


FOVER_PRM_MODELS = Literal["ryokamoi/Qwen-2.5-7B-FoVer-PRM",
                           "ryokamoi/Llama-3.1-8B-FoVer-PRM"]
baes_model_dict: dict[FOVER_PRM_MODELS, BASE_MODEL] = {
    "ryokamoi/Qwen-2.5-7B-FoVer-PRM": "Qwen/Qwen2.5-7B-Instruct",
    "ryokamoi/Llama-3.1-8B-FoVer-PRM": "meta-llama/Llama-3.1-8B-Instruct"
}

class QuickEvaluationTap(Tap):
    fover_prm_name: FOVER_PRM_MODELS = "ryokamoi/Qwen-2.5-7B-FoVer-PRM"
    """ The model name of FoVer-PRM to be evalauted. """
    dataset_dir: str = "quickstart/dataset/testdata"
    """ The path to the directory that includes dataset (test.jsonl) for evaluation. """
    output_dir: str = "quickstart/results"
    """ The directory to save the evaluation results. """


def main():
    args = QuickEvaluationTap().parse_args()
    print(f"Evaluating model: {args.fover_prm_name}")

    # this name will be used to save the performance
    dataset_short_name = Path(args.dataset_dir).stem

    # run verification
    arguments_list = [
        "--dataset_name", args.dataset_dir,
        "--base_model_name", baes_model_dict[args.fover_prm_name],
        "--verification_model_name", args.fover_prm_name,
        "--verification_prompt_type", "multi-turn",
    ]
    
    # run evaluation
    subprocess.run(["python", "src/direct_evaluation/run_direct_evaluation.py"] + arguments_list)
    
    # postprocess evaluation
    subprocess.run(["python", "src/direct_evaluation/postprocess_direct_evaluation.py"] + arguments_list)

    # save performance
    short_model_name = args.fover_prm_name.split("/")[-1]
    performance_dir = Path(args.output_dir) / dataset_short_name / short_model_name
    performance_dir.mkdir(parents=True, exist_ok=True)

    # copy evaluation metrics
    evaluation_metrics_path = get_direct_evaluation_metrics_path(
        dataset_name=args.dataset_dir,
        verification_model_name=args.fover_prm_name,
        split="test", verification_prompt_type="multi-turn",
    )
    shutil.copy(evaluation_metrics_path, performance_dir)


if __name__ == "__main__":
    main()

