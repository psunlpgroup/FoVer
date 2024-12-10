nli_labels = ["entailment", "contradiction", "neutral"]


def normalize_nli_final_answer(answer: str) -> str:
    """ Normalize nli final answer.
    We use this function for logicnli and anli."""
    
    for label in nli_labels:
        if label.lower() in answer.lower():
            return label
    
    raise ValueError(f"Invalid NLI answer: {answer}")


def normalize_hans_final_answer(answer: str) -> str:
    for label in ["non-entailment", "entailment"]:
        if label.lower() in answer.lower():
            return label
    
    raise ValueError(f"Invalid HANS answer: {answer}")
