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
In particular $2k = 0$.
Hence $k = 0$ and $n = 0$.
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
        "solution": """The sentence that best uses the possible options above for the first part is "In contrast to *Boycotts*, *Buycotts* aim to reward favourable behavior by companies."
The sentence that best uses the possible options above for the second part is "The success of such campaigns have been heightened through the use of *Digital technology*,"
The sentence that best uses the possible options above for the third part is "which allow campaigns to facilitate the company in achieving *Increased Sales*."
The answer is (F)""",
        "answer": "F"
    },
    {
        # question_id=18
        "problem": get_mmlu_instruction(
            question="Although the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:",
            options=[ "Online banners.", "Television advertising.", "Email marketing.", "Care lines.", "Direct mail.", "Inserts.", "Door to door.", "Radio advertising.", "Billboards.", "Social media advertising." ],
        ),
        "solution": """Door to door marketing delivers non-addressed items within all buildings within a geographic area.
While it can control the content and quality as well as direct mail marketing, its response rate is lower because of the lack of a personal address mechanism.
The answer is (G).""",
        "answer": "G"
    }
]


if __name__ == "__main__":
    # test the function
    for example in mmlu_pro_nomath_fewshot_prompt:
        print(example["problem"])
        print(example["solution"])
        print(example["answer"])
        print()
