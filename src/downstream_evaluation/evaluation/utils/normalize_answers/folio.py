folio_labels = ["True", "False", "Uncertain"]


def normalize_folio_final_answer(answer: str) -> str:
    """ Normalize FOLIO final answer. """
    
    for label in folio_labels:
        if label.lower() in answer.lower():
            return label
    
    raise ValueError(f"Invalid FOLIO response: {answer}")
