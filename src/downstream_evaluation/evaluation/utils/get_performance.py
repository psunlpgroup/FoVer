import numpy as np

from src.load_dataset import load_existing_dataset
from src.downstream_evaluation.evaluation.utils.extract_final_answer import extract_final_answer_for_downstream_evaluation
from src.downstream_evaluation.evaluation.utils.compare_final_answer import is_final_answer_for_downstream_evaluation_correct


def get_performance_for_downstream_evaluation(
        dataset_name: str, predictions: list[dict],
    ) -> dict:
    
    # get y_true
    dataset = load_existing_dataset(dataset_name)
    y_true = [example["y_true"] for example in dataset]
    
    # check ids
    if not all([dataset[idx]["id"] == predictions[idx]["id"]
            for idx in range(len(dataset))]):
        raise ValueError(
            f"""IDs do not match for the dataset: {dataset_name}
    y_true: {dataset[0]["id"]}, {dataset[1]["id"]}, ...
    y_pred: {predictions[0]["id"]}, {predictions[1]["id"]}, ...""")
    
    # get y_pred
    y_pred = [
        extract_final_answer_for_downstream_evaluation(
            dataset_name=dataset_name, prediction=prediction,
        )
        for prediction in predictions
    ]
    
    # get is_correct
    is_correct = [
        is_final_answer_for_downstream_evaluation_correct(
            dataset_name=dataset_name, y_true=y_true[idx], y_pred=y_pred[idx],
        )
        for idx in range(len(y_true))
    ]
    
    # get performance
    accuracy = np.mean(is_correct)
    
    return {
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "is_correct": is_correct,
    }
