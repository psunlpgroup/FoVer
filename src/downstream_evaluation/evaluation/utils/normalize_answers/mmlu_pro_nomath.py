def normalize_mmlu_pro_nomath_final_answer(answer: str) -> str:
    """Normalize the final answer for the AQuA-RAT dataset."""
    
    # all capital letters A-Z
    option_candidates = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    ]

    for option_candidate in option_candidates:
        if option_candidate in answer:
            return f"{option_candidate}"
    return answer  # not found
