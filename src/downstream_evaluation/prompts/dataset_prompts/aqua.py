# AQuA-RAT dataset
# few-shot examples are from train data and modified by us
# https://huggingface.co/datasets/deepmind

from src.downstream_evaluation.prompts.typing import FewShotPrompt


aqua_fewshot_prompt: list[FewShotPrompt] = [
    {
        "problem": """Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?
(A)21 (B)21.5 (C)22 (D)22.5 (E)23""",
        "solution": """If Q complete x kilometers, then P completes 1.15x kilometers.
The total distance is 43km, so we have: x + 1.15x = 43
2.15x = 43
x = 43 / 2.15 = 20
Q complete x kilometers, so P will have walked 1.15x = 1.15 * 20 = 23 km.
The answer is (E).""",
        "answer": "E"
    },
    {
        "problem": """In the coordinate plane, points (x, 1) and (5, y) are on line k. If line k passes through the origin and has slope 1/5, then what are the values of x and y respectively?
(A)4 and 1 (B)1 and 5 (C)5 and 1 (D)3 and 5 (E)5 and 3""",
        "solution": """"Line k passes through the origin and has slope 1/5" means that its equation is y = 1/5 * x.
Thus, for point (x, 1), we have 1 = 1/5 * x. Therefore, x = 5.
For point (5, y), we have y = 1/5 * 5 = 1.
Therefore, we have x = 5 and y = 1.
The answer is (C).""",
        "answer": "C"
    },
    {
        "problem": """There are k-2 members in a certain band, including Jim and Ellen. Two members are to be selected to attend the Grammy awards ceremony. If there are 6 possible combinations in which Jim and Ellen are not selected, what is the value of k?
(A)8 (B)9 (C)10 (D)11 (E)12""",
        "solution": """There are k-2 members in the band, and k-4 members without Jim and Ellen.
The number of ways to select 2 members from k-4 members is given by the combination formula: (k-4)C2 = (k-4)(k-5)/2.
We know that there are 6 possible combinations in which Jim and Ellen are not selected, so (k-4)C2 = 6.
So we can set up the equation: (k-4)(k-5) / 2 = 6.
(k-4)(k-5) = 12.
k^2 - 9k + 20 = 12.
k^2 - 9k + 8 = 0.
Factoring gives us (k-8)(k-1) = 0.
So k = 8 or k = 1.
There are k-2 members in the band, including Jim and Ellen, so k must be greater than 4. Therefore, we have k = 8.
Thus, the value of k is 8.
The answer is (A).""",
        "answer": "A"
    },
]
