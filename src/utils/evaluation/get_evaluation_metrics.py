import sklearn.metrics


def get_binary_evaluation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """ Get the binary evaluation metrics. """
    
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    return {
        "accuracy": sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
        "f1": f1, "precision": precision, "recall": recall,
    }


def get_float_evaluation_metrics(y_true: list[float], y_pred: list[float],
                                 threshold: float | None = None) -> dict:
    """ Get the float evaluation metrics. """

    metrics = {
        "auroc": sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred),
    }

    if threshold is not None:
        y_pred_binary = [pred > threshold for pred in y_pred]
        y_true_binary = [true > threshold for true in y_true]
        binary_metrics = get_binary_evaluation_metrics(
            y_true=y_true_binary, y_pred=y_pred_binary)

        metrics.update(binary_metrics)
        metrics["threshold"] = threshold
    
    return metrics


def get_threshold_for_f1(
        y_true: list[bool], y_pred: list[float],
) -> float:
    """ Get the threshold for f1 score. """
    
    best_f1 = 0
    best_threshold = 0
    thresholds = sorted(set(y_pred))
    for threshold in thresholds:
        y_pred_binary = [1 if pred > threshold else 0 for pred in y_pred]
        f1 = sklearn.metrics.f1_score(
            y_true=y_true, y_pred=y_pred_binary, zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
