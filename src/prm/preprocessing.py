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


ANSWER_START = "<ANSWER>"
ANSWER_END = "</ANSWER>"

reasoning_prompt_template = """** Problem **
{problem}

** Solution Steps **
{solution_steps}

** Instructions **
You are evaluating Step {step_number} out of {total_steps} shown above. Provide reasoning that determines whether this step is logically and mathematically correct given the problem and the context of the other steps. """ + f"""After you finish the reasoning, output a single line containing either {ANSWER_START}correct{ANSWER_END} if the step is valid or {ANSWER_START}incorrect{ANSWER_END} if you find an error. This line must be the final line of your response."""


def format_solution_steps(target_index: int, solution_steps: list[str]) -> str:
    formatted_steps: list[str] = []
    for idx, step in enumerate(solution_steps):
        label = f"* Step {idx + 1}"
        if idx == target_index:
            label += " (evaluate)"
        formatted_steps.append(f"** {label} **\n{step}")
    return "\n\n".join(formatted_steps)


def create_prompt_for_fover_with_reasoning(problem: str, target_step_idx: int, solution_steps: list[str], total_num_steps: int) -> str:
    return reasoning_prompt_template.format(
        problem=problem,
        solution_steps=format_solution_steps(target_step_idx, solution_steps),
        step_number=target_step_idx + 1,
        total_steps=total_num_steps,
    )


def get_fover_with_reasoning_input_format(
        problem: str, solution_steps: list[str],
        user_role_name: str = "user",
    ) -> list[list[dict]]:
    if len(solution_steps) == 0:
        return []

    total_steps = len(solution_steps)
    conversations: list[list[dict]] = []
    for idx in range(total_steps):
        prompt = create_prompt_for_fover_with_reasoning(
            problem=problem,
            target_step_idx=idx,
            solution_steps=solution_steps,
            total_num_steps=total_steps,
        )
        conversations.append([
            {
                "role": user_role_name,
                "content": prompt,
            }
        ])

    return conversations
