import json
from pathlib import Path
import random

from tap import Tap

from src.typing import BASE_MODEL
from src.config import splits_list
from src.path import get_initial_answers_path
from src.load_dataset import load_existing_dataset
from src.downstream_evaluation.evaluation.utils.extract_final_answer import \
    extract_final_answer_for_downstream_evaluation
from src.downstream_evaluation.evaluation.utils.compare_final_answer import \
    is_final_answer_for_downstream_evaluation_correct
from src.llm.utils import save_md5_hash


class IsabelleSelectInitialResponsesTap(Tap):
    model_name: str # BASE_MODEL
    dataset_name: str
    sample_k: int = 3
    prompt_type: str = "few-shot"
    ratio_of_correct_responses_in_selected: float = 0


def main():
    args = IsabelleSelectInitialResponsesTap().parse_args()
    
    for split in splits_list:
        all_initial_responses: list[dict[str, list[dict]]] = []
        
        dataset = load_existing_dataset(args.dataset_name, split=split)
        y_true = [example["y_true"] for example in dataset]
        
        id_to_data = {}
        for example in dataset:
            id_to_data[example["id"]] = example
        
        # classify responses into correct and incorrect
        stats_for_original_responses = {}
        for sample_idx in range(args.sample_k):
            initial_responses_path: Path = \
                get_initial_answers_path(
                    dataset_name=args.dataset_name, model_name=args.model_name,
                    split=split,
                    seed=sample_idx,
                )
            
            with open(initial_responses_path, "r") as f:
                initial_responses_list = [json.loads(line) for line in f]
            
            with open(initial_responses_path.with_suffix(
                    ".postprocessed.jsonl"), "r") as f:
                predictions = [json.loads(line) for line in f]
            
            # get y_pred
            y_pred = [
                extract_final_answer_for_downstream_evaluation(
                    dataset_name=args.dataset_name, prediction=prediction,
                )
                for prediction in predictions
            ]
            
            # get is_correct
            is_correct = [
                is_final_answer_for_downstream_evaluation_correct(
                    dataset_name=args.dataset_name,
                    y_true=y_true[idx], y_pred=y_pred[idx],
                )
                for idx in range(len(y_true))
            ]
            
            # initialize all_initial_responses
            if len(all_initial_responses) == 0:
                for idx in range(len(initial_responses_list)):
                    all_initial_responses.append(
                        {"y_correct": [], "y_incorrect": []}
                    )
            
            # append initial responses (attach y_pred and y_true)
            for idx in range(len(initial_responses_list)):
                # copy to avoid mutating the original loaded dict
                response_entry = dict(initial_responses_list[idx])
                
                # attach question
                response_entry["question"] = id_to_data[response_entry["id"]]["question"]
                
                # attach prediction and ground-truth for later use
                response_entry["y_pred"] = y_pred[idx]
                response_entry["y_true"] = y_true[idx]

                if is_correct[idx]:
                    all_initial_responses[idx]["y_correct"].append(
                        response_entry
                    )
                else:
                    all_initial_responses[idx]["y_incorrect"].append(
                        response_entry
                    )
            
            # save stats
            stats_for_original_responses[f"sample_{sample_idx}"] = {
                "total": len(initial_responses_list),
                "correct": sum(is_correct),
                "incorrect": len(initial_responses_list) - sum(is_correct),
            }
        
        # select incorrect responses as much as possible
        # from initial responses for each problem
        correct_responses = []
        incorrect_responses = []
        for idx, initial_responses in enumerate(all_initial_responses):
            if len(initial_responses["y_incorrect"]) > 0:
                selected = random.Random(idx).choice(
                    initial_responses["y_incorrect"]
                )
                selected["y_correct"] = False
                selected["question"] = id_to_data[selected["id"]]["question"]
                incorrect_responses.append(selected)
            else:
                selected = random.Random(idx).choice(
                    initial_responses["y_correct"]
                )
                selected["y_correct"] = True
                selected["question"] = id_to_data[selected["id"]]["question"]
                correct_responses.append(selected)
        
        # determine how many correct responses to include by refering to
        # args.ratio_of_correct_responses_in_selected
        target_number_of_correct_responses = int(
            len(incorrect_responses) * args.ratio_of_correct_responses_in_selected /
            (1 - args.ratio_of_correct_responses_in_selected)
        ) if args.ratio_of_correct_responses_in_selected > 0 else 0
        if len(correct_responses) > target_number_of_correct_responses:
            correct_responses = correct_responses[:target_number_of_correct_responses]
        else:
            raise ValueError(
                f"Not enough correct responses to achieve the target ratio "
                f"of {args.ratio_of_correct_responses_in_selected}: "
                f"have {len(correct_responses)}, "
                f"need {target_number_of_correct_responses}"
            )
        
        selected_responses = correct_responses + incorrect_responses
        selected_responses = random.Random(68).sample(
            selected_responses, len(selected_responses)
        )
        
        # save selected responses
        initial_responses_path = get_initial_answers_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=split, seed="selected"
        )
        initial_responses_path.parent.mkdir(parents=True, exist_ok=True)
        with open(initial_responses_path, "w") as f:
            for response in selected_responses:
                f.write(json.dumps(response) + "\n")
        save_md5_hash(initial_responses_path)
        
        # save statistics
        stats_path = initial_responses_path.with_suffix(".stats.json")
        stats = {
            "selected_responses": {
                "all": len(selected_responses),
                "correct": len(correct_responses),
                "incorrect": len(incorrect_responses),
            },
            "original_responses": stats_for_original_responses,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        
        print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
