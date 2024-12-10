from src.downstream_evaluation.prompts.typing import FewShotPrompt


# https://github.com/tommccoy1/hans/blob/master/heuristics_train_set.jsonl


def get_hans_instruction(premise: str, hypothesis: str) -> str:
    return f"""Premise: {premise}
Hypothesis: {hypothesis}
Question: do the premises derive the hypothesis? Provide reasoning and answer with either \"entailment\" or \"non-entailment\"."""


hans_fewshot_prompt: list[FewShotPrompt] = [
    {
        # ex24303
        "problem": get_hans_instruction(
            premise="Probably the scientist recommended the actors .",
            hypothesis="The scientist recommended the actors .",
        ),
        "solution": """Premise says, "Probably the scientist recommended the actors," which does not imply that the scientist recommended the actors.
The hypothesis states, "The scientist recommended the actors," which is not supported by the premise.
Therefore, the final answer is non-entailment.""",
        "answer": "non-entailment"
    },
    {
        # ex15281
        "problem": get_hans_instruction(
            premise="The secretary advised the manager and the judges .",
            hypothesis="The secretary advised the manager .",
        ),
        "solution": """Premise says, "The secretary advised the manager and the judges," which implies that the secretary advised the manager.
The hypothesis states, "The secretary advised the manager," which is supported by the premise.
Therefore, the final answer is entailment.""",
        "answer": "entailment"
    },
]
