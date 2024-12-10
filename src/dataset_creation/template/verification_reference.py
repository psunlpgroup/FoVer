step_template_start = """** Step {number} **
"""

cot_start_template = ""
# """Explanation:
#"""

step_template_end = """
<step_{number}>{label}</step_{number}>"""


verification_reference_template_with_cot = step_template_start + cot_start_template + "{explanation}" + step_template_end
verification_reference_template_no_cot = step_template_start + step_template_end


def get_verification_reference_for_single_turn_data(explanations_list: list[str], error_labels: list[bool]) -> str:
    """ Get the verification reference for the given explanations. """
    
    verification_reference_list: list[str] = []
    for idx, explanation in enumerate(explanations_list):
        label = "correct" if error_labels[idx] else "incorrect"
        
        if len(explanation) > 0:
            reference = verification_reference_template_with_cot.format(
                number=idx+1, explanation=explanation, label=label
            )
        else:
            reference = verification_reference_template_no_cot.format(
                number=idx+1, label=label
            )
        
        verification_reference_list.append(reference)
    
    return "\n\n".join(verification_reference_list)


from src.prm.preprocessing import get_fover_input_format
get_verification_reference_for_multi_turn_data = get_fover_input_format
