import re

from src.dataset_creation.base_dataset_specific.fol.typing import FOL_PROOF_LABEL


def split_fld_facts(facts: str) -> list[str]:
    """ Splits the given facts in the FLD dataset string into a list of facts.
    
    Args:
        facts (str): A string containing multiple facts.
    
    Returns:
        facts_list (list of str): A list of facts.
    """
    
    # Example fact1: (x): ¬{D}x -> ({A}x & {C}x) fact2: ¬({B}{a} & ¬{A}{a})"
    # Expected output: ["(x): ¬{D}x -> ({A}x & {C}x)", "¬({B}{a} & ¬{A}{a})"]
    facts_list = re.split(r"fact\d+: ", facts)[1:]
    
    # Remove the last space, if exists
    facts_list = [f.strip() for f in facts_list]
    
    return facts_list


def preprocess_fld_ground_truth_proof(proof_formula: str, proof_label: FOL_PROOF_LABEL) -> str:
    """ Preprocesses the ground truth proof in the FLD dataset. 
    * If the proof label is "DISPROVED", the hypothesis in the proof is replaced with "¬hypothesis".
        The original dataset contains "hypothesis" in the proof for "DISPROVED" proofs, which means "¬hypothesis".
    * replace "; " with ";\n"
    
    Args:
        proof_formula (str): The ground truth proof.
        proof_label (FOL_PROOF_LABEL): The proof label. One of "PROVED", "DISPROVED", or "UNKNOWN".
    
    Returns:
        proof_formula (str): The preprocessed proof.
    """
    
    if proof_label == "DISPROVED":
        # "hypothesis" -> "¬hypothesis"
        proof_formula = proof_formula.replace("hypothesis", "¬hypothesis")
    
    proof_formula = proof_formula.replace("; ", ";\n")
    
    return proof_formula


if __name__ == "__main__":
    # test code
    
    # facts
    facts = "fact1: (x): ¬{D}x -> ({A}x & {C}x) fact2: ¬({B}{a} & ¬{A}{a}) fact3: ¬{AB}{b} -> ¬{C}{d} fact4: ¬{C}{a} -> ¬{A}{d} fact5: ¬{A}{a} fact6: ({AB}{c} & ¬{B}{c}) -> {C}{d} fact7: ¬({F}{b} v {D}{b}) -> ¬{D}{a} fact8: {A}{a} -> ({AB}{c} & ¬{B}{c}) fact9: ({AB}{c} & ¬{AA}{c}) -> {B}{b} fact10: ({AB}{c} & ¬{B}{c}) -> {C}{b}"
    expected_output = ["(x): ¬{D}x -> ({A}x & {C}x)", "¬({B}{a} & ¬{A}{a})", "¬{AB}{b} -> ¬{C}{d}", "¬{C}{a} -> ¬{A}{d}", "¬{A}{a}", "({AB}{c} & ¬{B}{c}) -> {C}{d}", "¬({F}{b} v {D}{b}) -> ¬{D}{a}", "{A}{a} -> ({AB}{c} & ¬{B}{c})", "({AB}{c} & ¬{AA}{c}) -> {B}{b}", "({AB}{c} & ¬{B}{c}) -> {C}{b}"]
    facts_list = split_fld_facts(facts)
    assert all([f == e for f, e in zip(facts_list, expected_output)]), f"unexpected output: {facts_list}"
