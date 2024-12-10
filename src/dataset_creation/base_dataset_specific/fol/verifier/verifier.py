""" Verifier for FOL proofs. """


from typing import Optional

from src.dataset_creation.base_dataset_specific.fol.typing import FOL_PROOF_LABEL, FolStep, VERIFICATION_LABEL
from src.dataset_creation.base_dataset_specific.fol.utils import postprocess_fld_model_output, is_logically_equal

from FLD_generator.formula import Formula, negate, CONTRADICTION, CONJUNCTION
from FLD_generator.formula_checkers.z3_logic_checkers.checkers import check_sat


def verify_fol_one_step(fol_step: FolStep) -> VERIFICATION_LABEL:
    """
    Verifies if the given logical step is correct based on the provided facts.
    
    Args:
        fol_step (FolStep): A logical step to be verified, with assumptions and a conclusion.
    
    Returns:
        result (VERIFICATION_LABEL): A verification label for the step.
            One of "correct" or "wrong_implication"
    """
    
    assumptions = fol_step["assumptions"]
    conclusion = fol_step["conclusion"]
    
    # Create a solver and add assumptions to it:
    formula_reps: list[str] = []
    if type(assumptions) == str:
        formula_reps.append(Formula(assumptions))
    elif type(assumptions) == list:
        # this is not used in the main code
        
        if CONTRADICTION in assumptions:
            # If CONTRADICTION is among the assumptions, there should exists a contradiction.
            # This should only happen with assumptions, not intermediate steps.
            other_assumptions = [a for a in assumptions if a != CONTRADICTION]
            formula_reps.append(negate(Formula(f"{ CONJUNCTION }".join(other_assumptions))))
            
            # However, since we remove proof by contradiction from the dataset, this should not happen.
            raise ValueError("CONTRADICTION should not be among the assumptions. We removed proof by contradiction from the dataset.")
        else:
            # Otherwise, all assumptions should be true.
            for a in assumptions:
                formula_reps.append(Formula(a))
    else:
        raise ValueError("Unexpected type for assumptions:", type(assumptions))
    
    if conclusion == CONTRADICTION:
        # Check if assumptions are satisfiable or not:
        # This is often used to check if the assumptions are contradictory.
        sat = check_sat(formula_reps)
        
        # However, since we remove proof by contradiction from the dataset, this should not happen.
        raise ValueError("CONTRADICTION should not be the conclusion. We removed proof by contradiction from the dataset.")
        
        # return not sat
        
    else:  # conclusion is a logical expression (deduction step)
        # Check if facts and assumptions entail the conclusion:
        # This is true if facts & assumptions & ¬conclusion is unsatisfiable.
        formula_reps.append(negate(Formula(conclusion)))
        sat = check_sat(formula_reps)
            
        # return not sat
        if sat:
            return "wrong_implication"
        else:
            return "correct"


def verify_fol_steps_formula(steps: list[Optional[FolStep]]) -> list[VERIFICATION_LABEL]:
    """
    Verifies if the given logical steps are correct based on the provided facts.
    
    Args:
        facts (list of z3.ExprRef): List of Z3 logical expressions representing known facts.
        steps (list of FolStep): List of logical steps to be verified, with assumptions and conclusions.
            It may contain None for assumption steps, which will not be evaluated (assumed to be correct).
    
    Returns:
        result (list of VERIFICATION_LABEL): A list of verification labels for each step.
            One of "correct" or "wrong_implication"
    """
    
    result = []
    for step in steps:
        if step is None:  # assumption step
            # this can happen for proof by contradiction, but we removed it from the dataset
            # this will not happen in the current version of the dataset
            raise ValueError("Assumption steps are not expected in this version of the dataset.")
            result.append("assumption")
        else:
            result.append(verify_fol_one_step(step))
    
    return result


def verify_fld_proof_steps(model_output: str, facts_formula: str, hypothesis_formula: str, y_true: FOL_PROOF_LABEL) -> dict:
    """ Verifies the proof steps of FLD.
    
    Args:
        model_output (str): The output for the FLD dataset.
        facts_formula (str): The facts in the FLD dataset.
        hypothesis_formula (str): The hypothesis in the FLD dataset.
        y_true (FOL_PROOF_LABEL): The true proof label. One of "PROVED", "DISPROVED", or "UNKNOWN".
    
    Returns:
        result (dict): A dictionary containing the following
            - y_true (FOL_PROOF_LABEL): The true proof label.
            - y_pred (FOL_PROOF_LABEL): The predicted proof label.
            - y_correct (bool): True if the predicted proof label is correct.
            - proof_step_correctness (list of bool): A list of correctness for each proof step.
            - proof_step_labels (list of VERIFICATION_LABEL): A list of verification labels for each proof step.
                One of "correct" or "wrong_implication"
            - is_proof_consistent_to_y_pred (bool): True if the proof is consistent with the predicted proof label.
            - proof_steps (list of str): The proof steps in the original format.
            - processed_proof_steps (list of str): The processed proof steps.
    """
    
    # postprocess the model output
    y_pred, processed_steps, proof_steps = postprocess_fld_model_output(model_output, facts_formula, hypothesis_formula)
    
    # verify the proof steps
    result = verify_fol_steps_formula(processed_steps)
    
    # check the last step
    last_step = processed_steps[-1]["conclusion"]
    
    is_last_step_equal_to_hypothesis = is_logically_equal(Formula(last_step), Formula(hypothesis_formula))
    is_last_step_equal_to_negation_of_hypothesis = is_logically_equal(Formula(last_step), negate(Formula(hypothesis_formula)))
    
    is_proof_consistent_to_y_pred = True
    if y_pred == "PROVED":
        # the last conclusion should be "hypothesis"
        if not is_last_step_equal_to_hypothesis:
            is_proof_consistent_to_y_pred = False
    elif y_pred == "DISPROVED":
        # the last conclusion should be "¬hypothesis"
        if not is_last_step_equal_to_negation_of_hypothesis:
            is_proof_consistent_to_y_pred = False
    else:
        assert y_pred == "UNKNOWN"
        
        # the last conclusion should not be "hypothesis" or "¬hypothesis"
        if is_last_step_equal_to_hypothesis or is_last_step_equal_to_negation_of_hypothesis:
            is_proof_consistent_to_y_pred = False
    
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_correct": y_true == y_pred,
        "proof_step_correctness": [l == "correct" for l in result],
        "proof_step_labels": result,
        "is_proof_consistent_to_y_pred": is_proof_consistent_to_y_pred,
        "proof_steps": proof_steps,
        "processed_proof_steps": [f"""{folstep["assumptions"]} -> {folstep["conclusion"]}""" for folstep in processed_steps]
    }
