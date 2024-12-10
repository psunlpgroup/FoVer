import numpy as np

from src.utils.evaluation.get_evaluation_metrics import (
    get_binary_evaluation_metrics, get_float_evaluation_metrics
)


def accuracy_paired_bootstrap_test(
        y_correct_1: list[bool], y_correct_2: list[bool],
        sample_num: int=1000, seed: int=68) -> dict:
    """ Paired bootstrap test for accuracy. """

    if len(y_correct_1) != len(y_correct_2):
        raise ValueError(
            "y_correct_1 and y_correct_2 must have the same length."
        )
    
    num_1_is_better = 0
    for sample_idx in range(sample_num):
        # in paired bootstrap, we sample with replacement from both samples
        # with the same indices and compare the performance
        bootstrap_indices = np.random.RandomState(seed + sample_idx).choice(
            len(y_correct_1), len(y_correct_1), replace=True
        )
        sample_1 = [y_correct_1[idx] for idx in bootstrap_indices]
        sample_2 = [y_correct_2[idx] for idx in bootstrap_indices]
        
        accuracy_1 = np.mean(sample_1)
        accuracy_2 = np.mean(sample_2)
        
        # single-tailed test
        if accuracy_1 > accuracy_2:
            num_1_is_better += 1

    p_value = 1. - num_1_is_better / sample_num
    return {
        "p_value": p_value
    }


def float_scores_paired_bootstrap_test(
        y_pred_1: list[float], y_pred_2: list[float],
        y_true: list[bool],
        metric_name: str, threshold: float | None=None,
        sample_num: int=1000, seed: int=68) -> dict:
    """ Paired bootstrap test for AUROC. """

    if len(y_pred_1) != len(y_pred_2) or len(y_pred_1) != len(y_true):
        raise ValueError(
            "y_pred_1, y_pred_2 and y_true must have the same length."
        )

    num_1_is_better = 0
    for sample_idx in range(sample_num):
        # in paired bootstrap, we sample with replacement from both samples
        # with the same indices and compare the performance
        bootstrap_indices = np.random.RandomState(seed + sample_idx).choice(
            len(y_pred_1), len(y_pred_1), replace=True
        )
        sample_1 = [y_pred_1[idx] for idx in bootstrap_indices]
        sample_2 = [y_pred_2[idx] for idx in bootstrap_indices]
        y_true_sample = [y_true[idx] for idx in bootstrap_indices]
        
        if metric_name == "auroc":
            score_1 = get_float_evaluation_metrics(
                y_true=y_true_sample, y_pred=sample_1)["auroc"]
            score_2 = get_float_evaluation_metrics(
                y_true=y_true_sample, y_pred=sample_2)["auroc"]
        elif metric_name in ["f1", "precision", "recall"]:
            if threshold is None:
                raise ValueError(
                    "threshold must be specified for f1, precision and recall."
                )
            
            sample_1 = [1 if pred > threshold else 0 for pred in sample_1]
            sample_2 = [1 if pred > threshold else 0 for pred in sample_2]

            score_1 = get_binary_evaluation_metrics(
                y_true=y_true_sample, y_pred=sample_1)[metric_name]
            score_2 = get_binary_evaluation_metrics(
                y_true=y_true_sample, y_pred=sample_2)[metric_name]
        else:
            raise ValueError(
                f"metric_name must be one of "
                f"['auroc', 'f1', 'precision', 'recall'], "
                f"but got {metric_name}."
            )
        
        # single-tailed test
        if score_1 > score_2:
            num_1_is_better += 1
    
    p_value = 1. - num_1_is_better / sample_num
    return {
        "p_value": p_value
    }
