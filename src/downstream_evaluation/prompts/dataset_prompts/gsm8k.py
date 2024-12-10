# GSM8K
# https://github.com/kojima-takeshi188/zero_shot_cot/blob/5ef330fcdeec0cd26aee27943504f91f8ec1c33c/utils.py#L328
# reformulated to make post-processing easier

from src.downstream_evaluation.prompts.typing import FewShotPrompt


gsm8k_fewshot_prompt: list[FewShotPrompt] = [
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": """There were originally 15 trees in the grove.
After the grove workers planted trees today, there are now 21 trees.
So, the grove workers planted 21 - 15 = 6 trees today.
Therefore, the answer (arabic numerals) is 6.""",
        "answer": "6"
    },
    {
        "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "solution": """There were originally 3 cars in the parking lot.
2 more cars arrived, so there are now 3 + 2 = 5 cars in the parking lot.
Therefore, the answer (arabic numerals) is 5.""",
        "answer": "5"
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": """Originally, Leah had 32 chocolates and her sister had 42.
So, in total, they had 32 + 42 = 74 chocolates.
After eating 35, they had 74 - 35 = 39 left in total.
Therefore, the answer (arabic numerals) is 39.""",
        "answer": "39"
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": """Jason originally had 20 lollipops.
Then he had 12 after giving some to Denny.
So, he gave Denny 20 - 12 = 8 lollipops.
Therefore, the answer (arabic numerals) is 8.""",
        "answer": "8"
    },
]


gsm8k_fewshot_prompt_for_training_data: list[FewShotPrompt] = [
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": """There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
Therefore, the answer (arabic numerals) is 6.""",
        "answer": "6"
    },
    {
        "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "solution": """There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
Therefore, the answer (arabic numerals) is 5.""",
        "answer": "5"
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": """Originally, Leah had 32 chocolates. Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
Therefore, the answer (arabic numerals) is 39.""",
        "answer": "39"
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": """Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
Therefore, the answer (arabic numerals) is 8.""",
        "answer": "8"
    },
    {
        "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "solution": """Shawn started with 5 toys.
If he got 2 toys each from his mom and dad, then that is 4 more toys.
5 + 4 = 9.
Therefore, the answer (arabic numerals) is 9.""",
        "answer": "9"
    },
    {
        "problem": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "solution": """There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
9 + 20 is 29.
Therefore, the answer (arabic numerals) is 29.""",
        "answer": "29"
    },
    {
        "problem": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "solution": """Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33 golf balls.
Therefore, the answer (arabic numerals) is 33.""",
        "answer": "33"
    },
    {
        "problem": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "solution": """Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left.
23 - 15 is 8.
Therefore, the answer (arabic numerals) is 8.""",
        "answer": "8"
    },
]
