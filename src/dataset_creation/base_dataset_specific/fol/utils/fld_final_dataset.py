def get_fld_specific_items_in_final_dataset(
        error_labels_instance: dict[str, str]) -> dict:
    """ Get the verification data for the given error labels instance. """
    
    return {
        "hypothesis_formula": error_labels_instance["hypothesis_formula"],
        "facts_formula": error_labels_instance["facts_formula"],
    }
