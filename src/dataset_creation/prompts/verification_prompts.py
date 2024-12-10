# this prompt is only for single-turn data

""" This module includes the templates common to verification prompts for all tasks. This module is imported by prompt generation code for each task."""

import random


# all of the definitions provide the same information
# we use multiple definitions to increase the diversity of the prompts, which are generated using OpenAI o1 (Dec 22, 2024)
verification_task_definition_list = [
    """Your task is to review and critique each step of the solution.
For each, provide a brief explanation, and determine whether the reasoning and conclusion in that step are correct or incorrect.
If the step is correct, write `<step_{number}>correct</step_{number}>`. If the step is incorrect, write `<step_{number}>incorrect</step_{number}>`.
Make sure to replace {number} with the actual step number (for example, <step_1>, <step_2>, etc.).""",

    "**Your task** is to review and critique each step of the solution. "
"**For each step**, provide a brief explanation, and decide if the reasoning "
"and conclusion are correct or incorrect. **If the step is correct**, write "
"`<step_{number}>correct</step_{number}>`. **If the step is incorrect**, write "
"`<step_{number}>incorrect</step_{number}>`. Make sure to replace `{number}` with "
"the actual step (for example, `<step_1>`, `<step_2>`).",

    "You should *examine* every step of the solution and give a succinct explanation "
"of your judgment. Write `\"<step_{number}>correct</step_{number}>\"` if you decide "
"the step is valid, or `\"<step_{number}>incorrect</step_{number}>\"` if you find it invalid. "
"Replace `{number}` with the specific step index (such as `<step_1>`, `<step_2>`).",

    "Review the solution step by step. Provide a short explanation for each step. "
"Conclude whether the step’s logic and conclusion are correct or incorrect. "
"If correct, write `<step_{number}>correct</step_{number}>`. If incorrect, write "
"`<step_{number}>incorrect</step_{number}>`. Always substitute `{number}` with the "
"actual step number (for example, <step_1>, <step_2>, <step_3>).",

    "Your *task* is to **evaluate** each step in the solution. Provide a **short** "
"explanation of its correctness or incorrectness. If you find it correct, output "
"`<step_{number}>correct</step_{number}>`; if you find it *incorrect*, output "
"`<step_{number}>incorrect</step_{number}>`. Make sure to use the actual step number "
"(`<step_1>`, `<step_2>`, and so on) instead of `{number}`.",

    "Read through the solution, step by step (from start to finish). For each step, "
"offer a concise explanation of why it is valid or flawed. Mark valid steps with "
"`<step_{number}>correct</step_{number}>` and flawed steps with "
"`<step_{number}>incorrect</step_{number}>`. (Note: Replace `{number}` with `<step_1>`, "
"`<step_2>`, etc.)",

    "Inspect each step in the solution. Summarize *why* it is right or wrong. If it is "
"correct, output `<step_{number}>correct</step_{number}>`; if not, output "
"`<step_{number}>incorrect</step_{number}>`. **Do not forget** to replace `{number}` with "
"the actual step count (e.g., <step_1>, <step_2>).",

    """Your responsibility is to analyze each step.
---
State your brief reasoning for the correctness or incorrectness of that step.
---
If correct, include `<step_{number}>correct</step_{number}>`; if incorrect, include `<step_{number}>incorrect</step_{number}>`.
---
Replace `{number}` with the actual index, such as `<step_1>`, `<step_2>`. """,

    """Review each step in the solution one by one.
Explain your reasoning concisely for why you consider each step correct or incorrect.
If the step is correct, type `<step_{number}>correct</step_{number}>`.
If the step is incorrect, type `<step_{number}>incorrect</step_{number}>`.
Remember to replace `{number}` with the actual step index, like `<step_1>` or `<step_2>`.""",

    """> Carefully analyze each step of the solution.
> Offer a concise reason for your assessment, then label the step as correct or incorrect.
> If correct, write `<step_{number}>correct</step_{number}>`; if incorrect, write `<step_{number}>incorrect</step_{number}>`.
> Make sure to swap `{number}` for the right step identifier (e.g., `<step_1>`).""",

    """Analyze each step of the solution.
Decide whether it is correct or incorrect.
Offer a concise reason for that judgment.
If correct, write `<step_{number}>correct</step_{number}>`.
If incorrect, write `<step_{number}>incorrect</step_{number}>`.
Replace `{number}` with the corresponding step index, for example `<step_1>`, `<step_2>`. """,
]


# all of the definitions provide the same information
# we use multiple definitions to increase the diversity of the prompts, which are generated using GPT-4o (Dec 22, 2024)
verification_prompt = [
    """[Problem]
{problem}

[Solution]
{solution}

{task_definition}""",

    """Problem:
{problem}    

Solution:
{solution}

{task_definition}""",

    """In this task, you are tasked with evaluating the correctness of a solution. Carefully review the information provided and determine whether each step of the solution is correct.

Problem:
{problem}

Solution:
{solution}

Instructions:
{task_definition}""",

    """[Analysis Required]
Your task is to carefully review the details of the problem and the solution provided below.

Problem Statement:
{problem}

Solution Proposal:
{solution}

What to do:
{task_definition}""",

    """### Problem
{problem}

### Solution
{solution}

### Your Task
{task_definition}""",

    """Problem:
{problem}

Proposed Solution:
{solution}

Your Evaluation Task:
{task_definition}""",

    """1. Problem:
   {problem}

2. Solution:
   {solution}

Task Instructions:
{task_definition}""",

    """- Problem Description:
{problem}

- Solution Presented:
{solution}

What You Should Do:
{task_definition}""",

    """>>> Problem:
{problem}

>>> Solution:
{solution}

>>> Task:
{task_definition}"""
]


def get_randomly_selected_verification_prompt(data_id: str) -> tuple[str, str]:
    """ Get a randomly selected verification prompt for the given data ID. """
    
    # randomly select a definition and a prompt template
    task_definition = random.Random(f"{data_id}_definition").choice(verification_task_definition_list)
    prompt_template = random.Random(f"{data_id}_template").choice(verification_prompt)
    
    return prompt_template, task_definition


# Similar format to prompts in ProcessBench (Appendix D) https://arxiv.org/abs/2412.06559
# However, we ask to evaluate the correctness of the all steps in the proof.
step_format = """** Step {step_number} **
{step_content}"""


additional_instruction_for_not_trained_model = """* Your response should be in the format of '** Step 1 ** [explanation for Step 1] <step_1>correct</step_1> ** Step 2 ** [explanation for Step 2] <step_2>incorrect</step_2> ** Step 3 ** [explanation for Step 3] <step_3>correct</step_3> ...'
* Make sure to generate verifications for all steps in the Solution."""


def get_verification_prompt_for_single_turn_data(data_id: str, problem: str, solution_steps: list[str], using_not_trained_model: bool = False) -> str:
    # this is an old version

    """ Get a verification prompt for the given problem and solution steps.
    Templates and definitions are randomly selected for diversity, based on the data ID.
    
    using_not_trained_model: only used for evaluation on base models that are not fine-tuned on our datasets"""
    
    prompt_template, task_definition = get_randomly_selected_verification_prompt(data_id)
    
    solution_steps_content = "\n\n".join([step_format.format(step_number=i+1, step_content=step) for i, step in enumerate(solution_steps)])
    
    formatted_prompt = prompt_template.format(task_definition=task_definition, problem=problem, solution=solution_steps_content)
    
    if using_not_trained_model:  # only used for evaluation on base models that are not fine-tuned on our datasets
        formatted_prompt += "\n" + additional_instruction_for_not_trained_model
    return formatted_prompt
