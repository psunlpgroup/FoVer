from typing import TypedDict


class FewShotPrompt(TypedDict):
    problem: str
    solution: str  # solution with chain-of-thought
    answer: str  # short final answer
