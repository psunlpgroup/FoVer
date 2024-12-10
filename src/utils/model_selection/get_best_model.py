import json
from typing import Union

import numpy as np

from src.typing import BASE_MODEL, TRAIN_DATA, TRAIN_DATA_MULTI_TURN
from src.config import finetuned_verification_models_dict, \
    downstream_evaluation_for_model_selection_datasets_list, \
    train_dataset_names_list, train_dataset_names_list_multi_turn
from src.path import get_best_sample_and_rank_output_path, \
    get_downstream_evaluation_metrics_path


def get_full_list_of_verification_models(
        base_model_name: BASE_MODEL, train_data_name: TRAIN_DATA) -> list[str]:
    """ Get the full list of verification models for the given base model and
    train data. """
    
    full_list = [base_model_name]
    for verification_model_name_dict in finetuned_verification_models_dict[
            base_model_name].values():
        full_list.append(verification_model_name_dict[train_data_name])
    
    return full_list


def get_model_selection_performance_list(
            base_model_name: BASE_MODEL,
            verifier_names_list: list[str],
            verification_score_type: str,
            verification_prompt_type: str,
            metric_name: str="accuracy"
        ) -> tuple[list[float], dict[str, list[float]]]:
    """ Get the performance list of verification models for hyperparemeter
    selection. """
    
    # select the best verifier by the average validation performance
    verifier_performance_list: list[float] = []
    verifier_performance_list_per_dataset: dict[str, list[float]] = {}
    for verifier_name in verifier_names_list:
        # get the average performance of the verifier on the evaluation datasets
        metrics_list: list[float] = []
        for evaluation_dataset_name in \
                downstream_evaluation_for_model_selection_datasets_list:
            best_output_path = get_best_sample_and_rank_output_path(
                dataset_name=evaluation_dataset_name,
                base_model_name=base_model_name,
                verification_model_name=verifier_name,
                split="test", verification_prompt_type=verification_prompt_type,
                verification_score_type=verification_score_type,
            )
            metrics_path = get_downstream_evaluation_metrics_path(
                dataset_name=evaluation_dataset_name,
                model_name=base_model_name,
                prediction_path=best_output_path,
                split="test"
            )
            
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)[metric_name]
                metrics_list.append(metrics)
            else:
                print(f"Metrics file not found: {metrics_path}")
                metrics_list.append(-999999999)
                metrics = {}
            
            verifier_performance_list_per_dataset.setdefault(
                evaluation_dataset_name, []
            ).append(metrics)

        verifier_performance_list.append(np.mean(metrics_list).item())
    
    return verifier_performance_list, verifier_performance_list_per_dataset


def get_best_performance_verifier(
        base_model_name: BASE_MODEL,
        train_data_name: Union[TRAIN_DATA, TRAIN_DATA_MULTI_TURN],
        optimizer: str="AdamW",
        verification_score_type: str="logprob_min") -> str | None:
    """ Get verification model with the best performance in validation tasks."""
    
    if train_data_name not in finetuned_verification_models_dict[
            base_model_name][optimizer]:
        import warnings
        warnings.warn(f"Train data {train_data_name} not found.")
        return None
    else:
        if train_data_name in train_dataset_names_list:
            verification_prompt_type = "zero-shot"
        elif train_data_name in train_dataset_names_list_multi_turn:
            verification_prompt_type = "multi-turn"
        else:
            raise ValueError(f"Invalid train data name: {train_data_name}")
        
        candidates_list = finetuned_verification_models_dict[
            base_model_name][optimizer][train_data_name]

        if len(candidates_list) == 0:
            return None
        
        if len(candidates_list) == 1:
            return candidates_list[0]
        
        try:
            # if there are multiple candidates,
            # select the one with the best validation performance
            validation_performance_list, _ = \
                get_model_selection_performance_list(
                    base_model_name=base_model_name,
                    verifier_names_list=candidates_list,
                    verification_score_type=verification_score_type,
                    metric_name="accuracy",
                    verification_prompt_type=verification_prompt_type,
                )
            
            # get the index of the best verifier
            best_verifier_index = np.argmax(validation_performance_list)
            best_verifier_name = candidates_list[best_verifier_index]
        except Exception as e:
            print(f"Error in getting best verifier")
            import traceback
            traceback.print_exc()
            best_verifier_name = None
        
        return best_verifier_name
