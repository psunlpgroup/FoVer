def normalize_aqua_final_answer(answer: str) -> str:
    """Normalize the final answer for the AQuA-RAT dataset."""
    
    option_candidates = ["A", "B", "C", "D", "E"]
    for option_candidate in option_candidates:
        if option_candidate in answer:
            return f"{option_candidate}"
    return answer  # not found
