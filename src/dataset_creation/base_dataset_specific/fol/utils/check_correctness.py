""" This module contains functions for checking the correctness of the proof labels in the FLD dataset.
This code is for debugging purposes and not used in the dataset creation process. """


from src.dataset_creation.base_dataset_specific.fol.typing import FOL_PROOF_LABEL
from src.dataset_creation.base_dataset_specific.fol.utils.fld_preprocess import split_fld_facts

from FLD_generator.formula import Formula, negate
from FLD_generator.formula_checkers.z3_logic_checkers.checkers import check_sat


def get_fol_proof_label(facts: list[str], hypothesis: str) -> FOL_PROOF_LABEL:
    """ For a given list of facts and a hypothesis, verifies if the hypothesis can be proved or disproved based on the facts.
    This function is for checking the correctness of the original dataset and not used in the dataset creation process.
    
    Args:
        facts (list of str): List of logical expressions representing known facts.
        hypothesis (str): A logical expression representing the hypothesis.
    
    Returns:
        proof_label (FOL_PROOF_LABEL): A label indicating if the hypothesis can be proved, disproved, or is unknown.
    """
    
    # FLD can include random sentences, which are not related to the proof.
    if len(facts) == 1 and facts[0] == "fake_formula":
        return "UNKNOWN"
    facts = [f for f in facts if f != "fake_formula"]  # this case does not exists in the current version (0.31) of FLDx2
    
    try:
        # Check if hypothesis can be proved or disproved based on the facts:
        # This is true if facts & ¬hypothesis is unsatisfiable.
        sat = check_sat([Formula(f) for f in facts] + [negate(Formula(hypothesis))])
        
        if not sat:
            return "PROVED"
        
        # Check if hypothesis can be disproved based on the facts:
        # This is true if facts & hypothesis is unsatisfiable.
        sat = check_sat([Formula(f) for f in facts] + [Formula(hypothesis)])
        
        if not sat:
            return "DISPROVED"
    except Exception as e:
        print("Facts:", facts)
        print("Hypothesis:", hypothesis)
        
        raise e
    
    # If neither is true, the proof is unknown:
    return "UNKNOWN"


def is_proof_label_correct(fld_instance: dict[str, str]) -> bool:
    """ Checks if the proof label in the given FLD instance is correct.
    
    Args:
        fld_instance (dict): An instance of FLDx2.
    
    Returns:
        result (bool): True if the proof label is correct, False otherwise.
    """
    
    facts = split_fld_facts(fld_instance["facts_formula"])
    hypothesis = fld_instance["hypothesis_formula"]
    proof_label = fld_instance["proof_label"]
    
    # Check the correctness of the proof label:
    correct_proof_label = get_fol_proof_label(facts, hypothesis)
    
    return proof_label == correct_proof_label


if __name__ == "__main__":
    # test code
    
    # test case 1:
    fld_instance_1 = {
        "hypothesis_formula": "({AB}{c} & ¬{B}{c})",
        "facts_formula": "fact1: (x): ¬{D}x -> ({A}x & {C}x) fact2: ¬({B}{a} & ¬{A}{a}) fact3: ¬{AB}{b} -> ¬{C}{d} fact4: ¬{C}{a} -> ¬{A}{d} fact5: ¬{A}{a} fact6: ({AB}{c} & ¬{B}{c}) -> {C}{d} fact7: ¬({F}{b} v {D}{b}) -> ¬{D}{a} fact8: {A}{a} -> ({AB}{c} & ¬{B}{c}) fact9: ({AB}{c} & ¬{AA}{c}) -> {B}{b} fact10: ({AB}{c} & ¬{B}{c}) -> {C}{b}",
    }
    assert not is_proof_label_correct({**fld_instance_1, "proof_label": "PROVED"}), "unexpected output"
    assert not is_proof_label_correct({**fld_instance_1, "proof_label": "DISPROVED"}), "unexpected output"
    assert is_proof_label_correct({**fld_instance_1, "proof_label": "UNKNOWN"}), "unexpected output"
