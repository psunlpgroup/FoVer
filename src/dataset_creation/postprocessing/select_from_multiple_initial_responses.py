""" Select good initial responses from multiple initial responses. """

import json
import random

from src.typing import BASE_MODEL
from src.config import splits_list
from src.path import get_error_labels_path
from src.dataset_creation.initial_answer_generation.generate_initial_answers \
    import DatasetCreationTap
from src.dataset_creation.utils import get_error_labels_stats


class SelectBestFromMultipleInitialResponsesTap(DatasetCreationTap):
    model_name: str # BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def select_good_initial_responses_for_dataset_creation(
        all_initial_responses: list[list[dict]]):
    """ Select good initial responses from multiple initial responses. """
    
    ###
    # we need to preprocess all_initial_responses because they have different
    # number of responses as some of them are skipped due to verification
    # errors
    preprocessed_all_initial_responses: dict[str, list[dict]] = {}
    for initial_responses in all_initial_responses:
        for response in initial_responses:
            preprocessed_all_initial_responses.setdefault(
                response["id"], []).append(response)
    
    all_keys = sorted(list(preprocessed_all_initial_responses.keys()))
    
    ###
    # select the best initial response for each instance
    selected_responses_list = []
    for data_id in all_keys:
        initial_responses_for_instance = \
            preprocessed_all_initial_responses[data_id]

        ###
        # if all responses are correct, randomly select one of them
        if all(response["all_process_correct"] and response["y_correct"]
               for response in initial_responses_for_instance
        ):
            selected_responses_list.append(
                random.Random(data_id).choice(initial_responses_for_instance)
            )
            continue

        ###
        # in a specific probability, randomly select one of the initial
        # responses for each instance
        if random.Random(data_id).random() < 0.2:
            selected_responses_list.append(
                random.Random(data_id).choice(initial_responses_for_instance)
            )
            continue
        
        ###
        # otherwise, we only use responses with errors
        responses_with_errors = [
            response for response in initial_responses_for_instance
            if not response["all_process_correct"] or not response["y_correct"]
        ]
        
        # in a specific probability, randomly select one of the responses
        # that include errors
        if random.Random(data_id).random() < 0.8:
            selected_responses_list.append(
                random.Random(data_id).choice(responses_with_errors)
            )
            continue
        
        # otherwise select the responses with the smallerst number of errors
        # (we want to use challenging instances for training the verifier)
        selected_responses_list.append(
            min(responses_with_errors,
                key=lambda response:
                    response["proof_step_correctness"].count(False)
            )
        )
    
    return selected_responses_list


def main():
    args = SelectBestFromMultipleInitialResponsesTap().parse_args()
    
    for split in splits_list:
        all_initial_responses: list[list[dict]] = []
        for seed in range(1, args.num_samples + 1):
            # load error labels
            error_labels_path = get_error_labels_path(
                dataset_name=args.dataset_name, model_name=args.model_name,
                split=split, seed=seed
            )
            with open(error_labels_path, "r") as f:
                error_labels = [json.loads(line) for line in f]
            
            all_initial_responses.append(error_labels)
        
        # select challenging instances for training the verifier
        selected_responses = select_good_initial_responses_for_dataset_creation(
            all_initial_responses
        )
        
        # save selected responses
        selected_error_labels_path = get_error_labels_path(
            dataset_name=args.dataset_name, model_name=args.model_name,
            split=split, seed="selected"
        )
        selected_error_labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(selected_error_labels_path, "w") as f:
            for response in selected_responses:
                f.write(json.dumps(response) + "\n")
        
        # save stats
        stats = get_error_labels_stats(selected_responses)
        stats_path = selected_error_labels_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
