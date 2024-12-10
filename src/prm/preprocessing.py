""" You can use this input format for predicting step-level reward from our FoVer PRMs. Please refer to README.md for more details. """


tempalte_for_first_step_of_multi_turn_data = """** Problem **
{problem}

** Task **
Your task is to evaluate the accuracy of each step in the provided solution to the above question. For each step, respond with "correct" if the reasoning is logically valid and mathematically sound, or if the step is a general statement or transition that does not contain reasoning. Respond with "incorrect" if the step includes any errors or flawed logic.

** Sotluion **
{first_step}"""


def get_fover_input_format(
        problem: str, solution_steps: list[str],
        reference_error_labels: list[bool] | None = None,
        user_role_name = "user", model_role_name = "assistant",
    ) -> list[dict]:
    
    # make solution steps in string
    if reference_error_labels is None:
        # this is a dummy labels we use for inference
        reference_error_labels_str = ["correct"] * len(solution_steps)
    else:
        reference_error_labels_str = [
            "correct" if label else "incorrect"
            for label in reference_error_labels
        ]
    
    # make the first step
    first_step = tempalte_for_first_step_of_multi_turn_data.format(
        problem=problem, first_step=solution_steps[0]
    )
    
    conversation = [
        {
            "role": user_role_name,
            "content": first_step
        },
        {
            "role": model_role_name,
            "content": reference_error_labels_str[0]
        }
    ]
    
    # make the rest of the steps
    for idx, step in enumerate(solution_steps[1:], start=1):
        conversation.append(
            {
                "role": user_role_name,
                "content": step
            }
        )
        
        conversation.append(
            {
                "role": model_role_name,
                "content": reference_error_labels_str[idx]
            }
        )
    
    return conversation
