# MMLU Pro NoMath https://huggingface.co/blog/sam-paech/mmlu-pro-nomath

# few-shot examples are from the validation set
# we modified the solution to be suitable for our step-by-step format.
# https://huggingface.co/datasets/sam-paech/mmlu-pro-nomath-sml/viewer/default/validation

from src.downstream_evaluation.prompts.typing import FewShotPrompt


def get_mmlu_instruction(question: str, options: list[str]) -> str:
    # format of the options: (A) option1 (B) option2 (C) option3 ...
    options_str = "\n".join(
        [f"({chr(65 + i)}) {option}" for i, option in enumerate(options)]
    )

    # format the question
    return f"""{question}
{options_str}"""


mmlu_pro_nomath_fewshot_prompt: list[FewShotPrompt] = [
    {
        # question_id=0
        "problem": get_mmlu_instruction(
            question="The symmetric group $S_n$ has $actorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring 2Z.",
            options=[ "0", "30", "3", "10", "12", "50", "2", "100", "20", "5" ],
        ),
        "solution": """A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a \in 2Z$ implies that $k$ is a multiple of $n$.
Assume that $ka = 0$ for all $a \in 2Z$ for some $k$.
In particular, $2k = 0$.
Hence, $k = 0$ and $n = 0$.
The answer is (A).""",
        "answer": "A"
    },
    {
        # question_id=10
        "problem": get_mmlu_instruction(
            question="Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?",
            options=[ "1000 times more", "50 times more", "5000 times more", "500 times more", "10000 times more", "20000 times more", "2000 times more", "100 times more", "10 times more", "N/A" ],
        ),
        "solution": """The amount of light is proportional to the aperture area $A = \pi D^2/4$ for a lens with diameter $D$.
So the relative amounts of light between the eye with diameter 5mm and the telescope with diameter 50mm is $(50 cm)^2 / (5mm)^2 = 10000$.
The answer is (E).""",
        "answer": "E"
    },
    {
        # question_id=15
        "problem": get_mmlu_instruction(
            question="In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .",
            options=[ "Boycotts, Buyalls, Blockchain technology, Increased Sales", "Buycotts, Boycotts, Digital technology, Decreased Sales", "Boycotts, Buycotts, Digital technology, Decreased Sales", "Buycotts, Boycotts, Blockchain technology, Charitable donations", "Boycotts, Buyalls, Blockchain technology, Charitable donations", "Boycotts, Buycotts, Digital technology, Increased Sales", "Buycotts, Boycotts, Digital technology, Increased Sales", "Boycotts, Buycotts, Physical technology, Increased Sales", "Buycotts, Buyalls, Blockchain technology, Charitable donations", "Boycotts, Buycotts, Blockchain technology, Decreased Sales" ],
        ),
        "solution": """The sentence that best uses the available options for the first part, because it fits the contrast in the context, is "In contrast to *Boycotts*, *Buycotts* aim to reward favourable behavior by companies."
The sentence that best uses the available options for the second part, because it fits the context of how campaigns grow in effectiveness, is "The success of such campaigns have been heightened through the use of *Digital technology*,"
The sentence that best uses the available options for the third part, because it fits the intended positive business outcome in the context, is "which allow campaigns to facilitate the company in achieving *Increased Sales*."
Therefore, the best option is "Boycotts, Buycotts, Digital technology, Increased Sales".
The answer is (F)""",
        "answer": "F"
    },
    {
        # question_id=17
        "problem": get_mmlu_instruction(
            question="How can organisational structures that are characterised by democratic and inclusive styles of management be described?",
            options=["Flat", "Bureaucratic", "Autocratic", "Hierarchical", "Functional", "Decentralized", "Matrix", "Network", "Divisional", "Centralized"],
        ),
        "solution": """Option (A) Flat organisational structures have few management layers and promote shared decision making and open communication. Authority is distributed more evenly, supporting democratic and inclusive management. This matches the description in the question.
Option (B) Bureaucratic organisational structures rely on strict rules and formal hierarchies. Decision making is rigid and not inclusive. This does not match the description in the question.
Option (C) Autocratic organisational structures concentrate power in a single leader. Employee participation in decisions is minimal. This does not match the description in the question.
Option (D) Hierarchical organisational structures have many levels of authority with top down decision making. This limits inclusiveness. This does not match the description in the question.
Option (E) Functional organisational structures group employees by specialist roles. They focus on efficiency rather than inclusive decision making. This does not match the description in the question.
Option (F) Decentralized organisational structures spread decision making across units. They increase autonomy but do not necessarily ensure democratic management. This does not match the description in the question.
Option (G) Matrix organisational structures use dual reporting lines. They emphasize coordination rather than democratic leadership. This does not match the description in the question.
Option (H) Network organisational structures emphasize external partnerships and flexibility. They are not defined by inclusive internal management. This does not match the description in the question.
Option (I) Divisional organisational structures are organized by product or region. Authority remains structured within each division. This does not match the description in the question.
Option (J) Centralized organisational structures keep decision making at the top. Employee involvement is limited. This does not match the description in the question.
The answer is (A).""",
        "answer": "A"
    }
]


if __name__ == "__main__":
    # test the function
    for example in mmlu_pro_nomath_fewshot_prompt:
        print(example["problem"])
        print(example["solution"])
        print(example["answer"])
        print()
