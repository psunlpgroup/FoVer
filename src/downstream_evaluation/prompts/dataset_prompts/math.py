# MATH dataset
# prompt from https://arxiv.org/abs/2206.14858 (Appendix D.2)
# modifications:
# * we changed line breaks make them correspond to the sentence breaks
# * we removed "I hope it is correct." from the end of each solution
# * modified some solutions to make each sentence correspond to a reasoning step

from src.downstream_evaluation.prompts.typing import FewShotPrompt


math_fewshot_prompt: list[FewShotPrompt] = [
    {
        "problem": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": """The expressions inside each square root must be non-negative.
Therefore, $x-2 \\ge 0$, so $x \\ge 2$, and $5 - x \\ge 0$, so $x \\le 5$.
Also, the denominator cannot be equal to zero, so $5 - x > 0$, which gives $x < 5$.
Therefore, the domain of the expression is $[2,5)$.
The final answer is $[2,5)$.""",
        "answer": "$[2,5)$"
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": """We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = 24.$
The final answer is $24$.""",
        "answer": "$24$"
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": """If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2 \\cdot 12 \\cdot 20 = 480$ pounds of weight.
If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2 \\cdot 15 \\cdot n = 30n$ pounds of weight.
Equating this to 480 pounds, we can solve for $n$: $$30n &= 480 \\Rightarrow n &= 480/30 = 16$$
The final answer is $16$.""",
        "answer": "$16$"
    },
    {
        "problem": """If the system of equations
\\begin{align*}
6x-4y &= a, \\\\
6y-9x &= b.
\\end{align*}
has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.""",
        "solution": """If we multiply the first equation by $-\\frac{3}{2}$, we obtain $6y-9x = -\\frac{3}{2}a.$
Since we also know that $6y-9x = b$, we have $-\\frac{3}{2}a = b \\Rightarrow \\frac{a}{b} = -\\frac{2}{3}.$
The final answer is $-\\frac{2}{3}$.""",
        "answer": "$-\\frac{2}{3}$"
    }
]
