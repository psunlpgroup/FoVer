# few-shot examples made by the author of FoVer.

from src.downstream_evaluation.prompts.typing import FewShotPrompt


orca_math_fewshot_prompt: list[FewShotPrompt] = [
    {
        # bigmath_orca_math_063610
        "problem": "At its Grand Opening, the Guthrie Market distributed two different types of souvenirs. One type cost 20 cents each, and the other type cost a certain amount. One thousand souvenirs were distributed in all, and the cost of these souvenirs was 220 dollars. They distributed 400 of the souvenirs with the unknown cost. How much did this type of souvenir cost?",
        "solution": """400 of the souvenirs had a unknown cost.
That means the other 1000 - 400 = 600 souvenirs cost 20 cents each.
600 souvenirs at 20 cents each cost 600 * 20 = 12,000 cents.
The total cost was 220 dollars, which is 22,000 cents.
The 400 souvenirs cost 22,000 - 12,000 = 10,000 cents altogether.
Divide 10,000 by 400 to find the cost of each of those 400 souvenirs: 10,000 / 400 = 25
Each of the 400 souvenirs cost 25 cents.
Therefore, the answer (arabic numerals) is 25.""",
        "answer": "25"
    },
    {
        # bigmath_orca_math_063623
        "problem": "the average of 10 numbers is 40.2 . later it is found that two numbers have been wrongly copied . the first is 19 greater than the actual number and the second number added is 13 instead of 31 . find the correct average .",
        "solution": """The average of 10 numbers is 40.2, so their total is 40.2 * 10 = 402
One number is 19 too high, so subtract 19.
Another number is 18 too low (since 13 was used instead of 31), so add 18.
The corrected total is 402 - 19 + 18 = 401.
The correct average is 401 / 10 = 40.1.
Therefore, the answer (arabic numerals) is 40.1.""",
        "answer": "40.1"
    },
    {
        # bigmath_orca_math_063629
        "problem": "Every day, Lou works out by running three miles on a circular track that is one-quarter of a mile long and has a constant incline of 5%. His wife, Rosie, also runs on the same track at the same time as her husband, but with an adjusted pace. Due to the incline, Lou and Rosie slow down: Lou maintains 75% of his regular speed, while Rosie still runs at twice the speed of her husband, even on this inclined track. During their workout, how many times does Rosie circle the track?",
        "solution": """Lou runs 3 / 0.25 = 12 laps.
His speed is 0.75 * regular speed.
Rosie runs at 2 * Lou's speed on the track, so she runs at 2 * 0.75 = 1.5 * Lou's regular speed.
While Lou runs 3 miles, Rosie runs 3 * 1.5 = 4.5 miles.
Rosie circles the track 4.5 / 0.25 = 18 times.
Therefore, the answer (arabic numerals) is 18.""",
        "answer": "18"
    },
]
