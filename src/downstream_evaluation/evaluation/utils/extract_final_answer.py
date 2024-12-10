from src.config import get_downstream_evaluation_datasets_lsit


def extract_from_box(answer: str) -> str | None:
    """Extract the answer from the box."""
    
    # get \\boxed{answer}
    prefix = "\\boxed{"
    start_index = answer.find(prefix) + len(prefix)
    end_index = answer.rfind("}")
    
    if start_index - len(prefix) == -1 or end_index == -1:
        return None
    else:
        return answer[start_index:end_index]


full_downstream_evaluation_datasets_list = \
    get_downstream_evaluation_datasets_lsit("model_selection") + \
    get_downstream_evaluation_datasets_lsit("final_evaluation") + \
    ["bigmath_math_word_problems", "metamathqa_gsm8k"]

def extract_final_answer_for_downstream_evaluation(
        dataset_name: str, prediction: dict) -> str | None:
    """Extract the final answer from the response
    for the downstream evaluation."""
    
    # already preprocessed
    # this happens for self-consistency
    if "y_pred" in prediction.keys():
        return prediction["y_pred"]

    # get $\\boxed{answer}$
    extracted_string = extract_from_box(prediction["response"])
    
    # postprocess
    if dataset_name in full_downstream_evaluation_datasets_list:
        try:
            if dataset_name in ["gsm8k", "metamathqa_gsm8k", "aime"]:
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.gsm8k import normalize_gsm8k_final_answer
                return normalize_gsm8k_final_answer(
                    extracted_string, cast="int"
                )
            elif dataset_name in ["bigmath_math_word_problems", "orca_math"]:
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.gsm8k import normalize_gsm8k_final_answer
                return normalize_gsm8k_final_answer(
                    extracted_string, cast="float"
                )
            elif dataset_name == "math":
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.math import normalize_math_final_answer
                return normalize_math_final_answer(extracted_string)
            elif dataset_name == "aqua":
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.aqua import normalize_aqua_final_answer
                return normalize_aqua_final_answer(extracted_string)
            elif "bbh" in dataset_name:
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.bbh import normalize_bbh_final_answer
                return normalize_bbh_final_answer(
                    extracted_string, dataset_name=dataset_name
                )
            elif dataset_name == "folio":
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.folio import normalize_folio_final_answer
                return normalize_folio_final_answer(extracted_string)
            elif dataset_name in ["logicnli", "anli"]:
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.nli import normalize_nli_final_answer
                return normalize_nli_final_answer(extracted_string)
            elif dataset_name == "hans":
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.nli import normalize_hans_final_answer
                return normalize_hans_final_answer(extracted_string)
            elif dataset_name == "mmlu_pro_nomath":
                from src.downstream_evaluation.evaluation.utils.\
                    normalize_answers.mmlu_pro_nomath \
                    import normalize_mmlu_pro_nomath_final_answer
                return normalize_mmlu_pro_nomath_final_answer(extracted_string)
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
        except Exception as e:
            print("Error in extract_final_answer_for_downstream_evaluation")
            print(f"\tDataset name: {dataset_name}")
            print(f"\tPrediction: {prediction}")
            print(f"\tError: {e}")
            return None
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
