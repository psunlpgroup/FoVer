def normalize_bbh_final_answer(answer: str, dataset_name: str) -> str:
    """Normalize the final answer for the BBH dataset."""
    
    # BBH dataset
    if dataset_name == "bbh_word_sorting":
        return answer
    elif "bbh_tracking_shuffled_objects" in dataset_name \
            or dataset_name == "bbh_temporal_sequences" \
            or dataset_name == "bbh_logical_deduction_three_objects" :
        # bbh_tracking_shuffled_objects_three_objects: A, B, C
        # bbh_temporal_sequences: A, B, C, D
        option_candidates = ["A", "B", "C", "D"]
        for option_candidate in option_candidates:
            if option_candidate in answer:
                return f"({option_candidate})"
        return answer  # not found
    elif dataset_name == "bbh_formal_fallacies":
        option_candidates = ["valid", "invalid"]
        for option_candidate in option_candidates:
            if option_candidate in answer.lower():
                return option_candidate
        return answer  # not found
    # elif dataset_name == "bbh_navigate":
    #     option_candidates = ["Yes", "No"]
    #     for option_candidate in option_candidates:
    #         if option_candidate in answer:
    #             return option_candidate
    #     return answer  # not found
    elif dataset_name == "bbh_boolean_expressions":
        option_candidates = ["True", "False"]
        for option_candidate in option_candidates:
            if option_candidate in answer:
                return option_candidate
        return answer
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
