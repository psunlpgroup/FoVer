import re
import copy

from src.dataset_creation.base_dataset_specific.fol.typing import FOL_PROOF_LABEL, FolStep


def postprocess_fld_model_output(model_output: str, facts_formula: str, hypothesis_formula: str) -> tuple[FOL_PROOF_LABEL, list[FolStep], list[str]]:
    """ Post-processes the output of the FLD model to extract the logical steps and the proof.
    
    facts_formula "fact1: \u00ac{M} -> ({F} & {I})\nfact2: {T} -> \u00ac(\u00ac{N} & \u00ac{L})\nfact3: \u00ac(\u00ac{Q} & \u00ac{R}) -> {P}\nfact4: \u00ac{U} -> \u00ac(\u00ac{AA} & \u00ac{T})\nfact5: {IL}\nfact6: (\u00ac{B} & \u00ac{A}) -> {G}\nfact7: {B}\nfact8: \u00ac{U}\nfact9: \u00ac{E} -> ({D} & {C})\nfact10: (\u00ac{J} & {K}) -> {H}\nfact11: {A}\nfact12: ({H} & {F}) -> \u00ac{E}\nfact13: {P} -> (\u00ac{M} & {O})\nfact14: \u00ac(\u00ac{AA} & \u00ac{T}) -> {T}\nfact15: {C} -> (\u00ac{B} & \u00ac{A})\nfact16: \u00ac(\u00ac{N} & \u00ac{L}) -> {L}\nfact17: {L} -> (\u00ac{J} & {K})\nfact18: ({E} & {EH})\nfact19: \u00ac({D} v \u00ac{E}) -> \u00ac{C}"
    hypothesis_formula "({A} & {B})"
    
    Model responses format "$proof$:\nfact1 & fact12 -> int1: (\u00ac{AA}{b} v \u00ac{AB}{b});\nfact2 & int1 -> int2: \u00ac(\u00ac{AA}{b} v \u00ac{AB}{b});\nint2 -> hypothesis;\n\n$proof_label$: PROVED"
    
    Args:
        model_output (str): The output of the FLD model.
        facts (str): The facts in the FLD dataset.
        hypothesis (str): The hypothesis in the FLD dataset.
    
    Returns:
        y_pred (FOL_PROOF_LABEL): The predicted proof label. One of "PROVED", "DISPROVED", or "UNKNOWN".
        processed_steps (list of FolStep): A list of logical steps extracted from the model output.
        proof_steps (list of str): A list of original steps extracted from the model output.
    """
    
    # remove "$proof$:"
    # it also can remove unnecessary text before the proof (it rarely happens)
    model_output = model_output.split("$proof$:\n", maxsplit=1)[1]
    
    # split into proof and label
    proof_label_str = "\n\n$proof_label$: "
    if proof_label_str not in model_output:
        # output is too long and truncated during generation
        raise ValueError("Model output is too long and truncated during generation.")
    
    proof_string, proof_label = model_output.split(proof_label_str)
    
    # if the proof is empty, return None
    if len(proof_string) == 0:
        raise ValueError("Proof is empty.")
    
    # remove unnecessary text from proof_label
    for proof_label_candidate in reversed(FOL_PROOF_LABEL.__args__):  # iterate over all the possible proof labels
        if proof_label_candidate in proof_label:
            proof_label = proof_label_candidate
            break
    proof_label: FOL_PROOF_LABEL = proof_label
    
    # split proof into steps
    proof_steps = proof_string.split("\n")
    proof_steps = [step[:-1] if step[-1] == ";" else step for step in proof_steps]  # remove the last semicolon
    
    ###
    # Replace facts, intermediate value, and hypothesis with their actual values
    
    # make a dictionary of facts
    facts_replacement_list = []
    for idx, fact in enumerate(facts_formula.split("\n")):
        key = f"fact{idx + 1}"
        value = fact[len(key) + 2:]  # remove "fact{idx + 1}: "
        facts_replacement_list.append((key, f"({value})"))

    # reverse the list to replace in the correct order
    # for example, "fact12" should be replaced before "fact1"
    facts_replacement_list = list(reversed(facts_replacement_list))
    
    # make a dictionary of intermediate values
    # "fact1 & fact12 -> int1: (\u00ac{AA}{b} v \u00ac{AB}{b})" -> {"int1": "(\u00ac{AA}{b} v \u00ac{AB}{b})"}
    intermediate_replacement_list = []
    steps_premise_conclusion_list: list[tuple[str, str]] = []
    for step in proof_steps:
        step = copy.deepcopy(step)
        premise, conclusion = step.split(" -> ", maxsplit=1)

        # detect starting idx of "int{idx}: " using re
        # it is possible that the intermediate value is not used in the proof
        match = re.search(r"int\d+: ", conclusion)
        if match:
            start_idx = match.start()
            end_idx = match.end()
            assert start_idx == 0, f"unexpected start_idx: {start_idx} for step: {step}"
            
            # add the intermediate value to the replacement list
            intermediate_replacement_list.append((conclusion[start_idx:end_idx-2], f"({conclusion[end_idx:]})"))
            
            # remove "int\d+: " from the step
            conclusion = conclusion[end_idx:]
        
        # split the step into premise and conclusion
        steps_premise_conclusion_list.append((premise, conclusion))
    
    # reverse the list to replace in the correct order
    # for example, "int11" should be replaced before "int1"
    intermediate_replacement_list = list(reversed(intermediate_replacement_list))
    
    # replace intermediate values with their actual values
    replacement_list = facts_replacement_list + list(intermediate_replacement_list) + [("hypothesis", f"({hypothesis_formula})")]
    
    processed_steps: list[FolStep] = []
    for idx, (premise, conclusion) in enumerate(steps_premise_conclusion_list):
        for key, value in replacement_list:
            premise = premise.replace(key, value)
            conclusion = conclusion.replace(key, value)
        
        processed_steps.append(
            FolStep(
                assumptions=premise,
                conclusion=conclusion
            )
        )
    
    return proof_label, processed_steps, proof_steps


if __name__ == "__main__":
    # test case
    example_fact_formula = "fact1: \u00ac{M} -> ({F} & {I})\nfact2: {T} -> \u00ac(\u00ac{N} & \u00ac{L})\nfact3: \u00ac(\u00ac{Q} & \u00ac{R}) -> {P}\nfact4: \u00ac{U} -> \u00ac(\u00ac{AA} & \u00ac{T})\nfact5: {IL}\nfact6: (\u00ac{B} & \u00ac{A}) -> {G}\nfact7: {B}\nfact8: \u00ac{U}\nfact9: \u00ac{E} -> ({D} & {C})\nfact10: (\u00ac{J} & {K}) -> {H}\nfact11: {A}\nfact12: ({H} & {F}) -> \u00ac{E}\nfact13: {P} -> (\u00ac{M} & {O})\nfact14: \u00ac(\u00ac{AA} & \u00ac{T}) -> {T}\nfact15: {C} -> (\u00ac{B} & \u00ac{A})\nfact16: \u00ac(\u00ac{N} & \u00ac{L}) -> {L}\nfact17: {L} -> (\u00ac{J} & {K})\nfact18: ({E} & {EH})\nfact19: \u00ac({D} v \u00ac{E}) -> \u00ac{C}"
    example_hypothesis_formula = "({A} & {B})"
    example_model_output = "$proof$:\nfact1 & fact12 -> int1: (\u00ac{AA}{b} v \u00ac{AB}{b});\nfact2 & int1 -> int2: \u00ac(\u00ac{AA}{b} v \u00ac{AB}{b});\nint2 -> hypothesis;\n\n$proof_label$: PROVED"
    
    proof_label, processed_steps, proof_steps = postprocess_fld_model_output(model_output=example_model_output, facts_formula=example_fact_formula, hypothesis_formula=example_hypothesis_formula)

    print("proof_label:", proof_label)
    
    print("proof_steps:")
    for step in proof_steps:
        print(step)
    
    print("processed_steps:")
    for step in processed_steps:
        print(step["assumptions"], "     ->     ", step["conclusion"])
