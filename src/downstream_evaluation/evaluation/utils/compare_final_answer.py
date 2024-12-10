def is_final_answer_for_downstream_evaluation_correct(
        dataset_name: str, y_true: str | None, y_pred: str | None) -> bool:
    """Check if the final answer is correct."""
    
    if y_true is None or y_pred is None:
        return False
    
    if dataset_name in ["gsm8k", "metamathqa_gsm8k", "aime"]:
        from src.downstream_evaluation.evaluation.utils.normalize_answers.\
            gsm8k import string_to_int
        return string_to_int(y_true) == string_to_int(y_pred)
    elif dataset_name in ["orca_math", "bigmath_math_word_problems"]:
        import numpy as np
        from src.downstream_evaluation.evaluation.utils.normalize_answers.\
            gsm8k import string_to_float

        return bool(
            np.isclose(
                string_to_float(y_true), string_to_float(y_pred), atol=1e-5
            )
        )
    elif "bbh" in dataset_name:
        # if y_true is included in y_pred, then it is regarded as correct
        # this is for bbh_word_sorting task
        # for other tasks, we can use y_true == y_pred
        return y_true in y_pred
    elif dataset_name in ["math", "aqua", "folio", "logicnli", "anli", "hans",
                          "mmlu_pro_nomath"]:
        return y_true == y_pred  # already preprocessed
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
