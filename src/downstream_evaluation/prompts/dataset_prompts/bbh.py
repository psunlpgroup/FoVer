from src.downstream_evaluation.prompts.typing import FewShotPrompt


# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/word_sorting.txt
bbh_word_sorting_prompt: list[FewShotPrompt] = [
    {
        "problem": "Sort the following words alphabetically: List: oven costume counterpart",
        "solution": """First, we look at the first letters of each word. "oven": "o" (15th letter). "costume": "c" (3rd letter). "counterpart": "c" (3rd letter). We now have: ["costume" "counterpart"] < "oven".
Now let's sort this subpart ["costume" "counterpart"] by looking at their second letters. The second letter: "costume": "o" (15th letter). "counterpart": "o" (15th letter). They are equal, so we need to look at the third letters.
Now let's sort this subpart ["costume" "counterpart"] by looking at their third letters. The third letter: "costume": "s" (19 th letter). "counterpart": "u" (21st letter). We now have: "costume" < "counterpart" < "oven".
So the answer is costume < counterpart < oven.""",
        "answer": "costume counterpart oven"
    },
    {
        "problem": "Sort the following words alphabetically: List: hypochlorite ponderosa phone credulity",
        "solution": """First, we look at the first letters of each word. "hypochlorite": "h" (8th letter). "ponderosa": "p" (16th letter). "phone": "p" (16th letter). "credulity": "c" (3rd letter). We now have: "credulity" < "hypochlorite" < ["ponderosa" "phone"].
Now let's sort this subpart ["ponderosa" "phone"] by looking at their second letters. The second letter: "ponderosa": "o" (15th letter). "phone": "h" (8th letter). We now have:  "credulity" < "hypochlorite" < "phone" < "ponderosa".
So the answer is credulity < hypochlorite < phone < ponderosa.""",
        "answer": "credulity hypochlorite phone ponderosa"
    },
    {
        "problem": "Sort the following words alphabetically: List: newt arson parthia seismography mugho aspect census",
        "solution": """First, we look at the first letters of each word. "newt": "n" (14th letter). "arson": "a" (1st letter). "parthia": "p" (16th letter). "seismography": "s" (19th letter). "mugho": "m" (13th letter). "aspect": "a" (1st letter). "census": "c" (3rd letter). We now have: ["arson" "aspect"] < "census" < "mugho" < "newt" < "parthia" < "seismography".
Now let's sort this subpart ["arson" "aspect"] by looking at their second letters. The second letter: "arson": "r" (18th letter). "aspect": "s" (19th letter). We now have: "arson" < "aspect" < "census" < "mugho" < "newt" < "parthia" < "seismography".
So the answer is arson < aspect < census < mugho < newt < parthia < seismography.""",
        "answer": "arson aspect census mugho newt parthia seismography"
    }
]


# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/tracking_shuffled_objects_three_objects.txt
# reasoning steps are updated by the fover authors
bbh_tracking_shuffled_objects_three_objects: list[FewShotPrompt] = [
    {
        "problem": """Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a yellow ball, Bob has a blue ball, and Claire has a pink ball.
As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Alice and Bob swap balls. Finally, Claire and Bob swap balls. At the end of the game, Bob has the
Options:
(A) yellow ball
(B) blue ball
(C) pink ball""",
        "solution": """At the start, Alice has a yellow ball, Bob has a blue ball, and Claire has a pink ball.
First, Claire and Alice swap balls. Now, Alice has a pink ball, Bob has a blue ball, and Claire has a yellow ball.
Then, Alice and Bob swap balls. Now, Alice has a blue ball, Bob has a pink ball, and Claire has a yellow ball.
Finally, Claire and Bob swap balls. Now, Alice has a blue ball, Bob has a yellow ball, and Claire has a pink ball.
At the end of the game, Bob has the yellow ball.
So the answer is (A).""",
        "answer": "A"
    },
    {
        "problem": """Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a white ball, Bob has a purple ball, and Claire has a pink ball.
As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Bob and Claire swap balls. Finally, Bob and Alice swap balls. At the end of the game, Alice has the
Options:
(A) white ball
(B) purple ball
(C) pink ball""",
        "solution": """At the start, Alice has a white ball, Bob has a purple ball, and Claire has a pink ball.
First, Bob and Alice swap balls: Alice has a purple ball, Bob has a white ball, and Claire has a pink ball.
Then, Bob and Claire swap balls: Alice has a purple ball, Bob has a pink ball, and Claire has a white ball.
Finally, Bob and Alice swap balls: Alice has a pink ball, Bob has a purple ball, and Claire has a white ball.
At the end of the game, Alice has the pink ball.
So the answer is (C).""",
        "answer": "C"
    },
    {
        "problem": """Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Lola, Bob is dancing with Rodrigo, and Claire is dancing with Patrick.
Throughout the song, the dancers often trade partners. First, Alice and Bob switch partners. Then, Claire and Bob switch partners. Finally, Bob and Alice switch partners. At the end of the dance, Alice is dancing with
Options:
(A) Lola
(B) Rodrigo
(C) Patrick""",
        "solution": """"At the start of the dance, Alice is dancing with Lola, Bob is dancing with Rodrigo, and Claire is dancing with Patrick.
First, Alice and Bob switch partners: Alice is dancing with Rodrigo, Bob is dancing with Lola, and Claire is dancing with Patrick.
Then, Claire and Bob switch partners: Alice is dancing with Rodrigo, Bob is dancing with Patrick, and Claire is dancing with Lola.
Finally, Bob and Alice switch partners: Alice is dancing with Patrick, Bob is dancing with Rodrigo, and Claire is dancing with Lola.
At the end of the dance, Alice is dancing with Patrick.
So the answer is (C).""",
        "answer": "C"
    }
]


# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/temporal_sequences.txt
# reasoning steps are updated by the fover authors
bbh_temporal_sequences: list[FewShotPrompt] = [
    {
        "problem": """Today, Emily went to the museum. Between what times could they have gone?
We know that:
Emily woke up at 1pm.
Elizabeth saw Emily reading at the library from 2pm to 4pm.
Jessica saw Emily watching a movie at the theater from 4pm to 5pm.
Leslie saw Emily waiting at the airport from 5pm to 6pm.
William saw Emily buying clothes at the mall from 6pm to 7pm.
The museum was closed after 7pm.
Between what times could Emily have gone to the museum?
Options:
(A) 1pm to 2pm
(B) 6pm to 7pm
(C) 5pm to 6pm
(D) 2pm to 4pm""",
        "solution": """Emily wake up at 1pm, so she did not go to the museum before 1pm.
Emily wake up at 1pm, and Elizabeth saw Emily reading at the library from 2pm to 4pm, so Emily could have gone to the museum from 1pm to 2pm.
Elizabeth saw Emily reading at the library from 2pm to 4pm, so Emily did not go to the museum from 2pm to 4pm.
Jessica saw Emily watching a movie at the theater from 4pm to 5pm, so Emily did not go to the museum from 4pm to 5pm.
Leslie saw Emily waiting at the airport from 5pm to 6pm, so Emily did not go to the museum from 5pm to 6pm.
William saw Emily buying clothes at the mall from 6pm to 7pm, so Emily did not go to the museum from 6pm to 7pm.
The museum was closed after 7pm, so Emily did not go to the museum after 7pm.
Im summary, the only time when Emily could have gone to the museum was 1pm to 2pm.
From the options, we can see that 1pm to 2pm is the only option that fits this time frame.
So the answer is (A).""",
        "answer": "A"
    },
    {
        "problem": """Today, Elizabeth went to the amusement park. Between what times could they have gone?
We know that:
Elizabeth woke up at 7am.
David saw Elizabeth fixing their computer at the electronic store from 1pm to 2pm.
Sarah saw Elizabeth playing tennis at the tennis court from 2pm to 3pm.
Susan saw Elizabeth walking towards the Statue of Liberty from 3pm to 6pm.
Andrew saw Elizabeth taking photos near the Eiffel Tower from 6pm to 9pm.
Emily saw Elizabeth getting a coffee at the cafe from 9pm to 10pm.
The amusement park was closed after 10pm.
Between what times could Elizabeth have gone to the amusement park?
Options:
(A) 7am to 1pm
(B) 9pm to 10pm
(C) 1pm to 2pm
(D) 3pm to 6pm""",
        "solution": """Elizabeth wake up at 7am, so she did not go to the amusement park before 7am.
Elizabeth wake up at 7am, and David saw Elizabeth fixing their computer at the electronic store from 1pm to 2pm, so Elizabeth could have gone to the amusement park from 7am to 1pm.
David saw Elizabeth fixing their computer at the electronic store from 1pm to 2pm, so Elizabeth did not go to the amusement park from 1pm to 2pm.
Sarah saw Elizabeth playing tennis at the tennis court from 2pm to 3pm, so Elizabeth did not go to the amusement park from 2pm to 3pm.
Susan saw Elizabeth walking towards the Statue of Liberty from 3pm to 6pm, so Elizabeth did not go to the amusement park from 3pm to 6pm.
Andrew saw Elizabeth taking photos near the Eiffel Tower from 6pm to 9pm, so Elizabeth did not go to the amusement park from 6pm to 9pm.
Emily saw Elizabeth getting a coffee at the cafe from 9pm to 10pm, so Elizabeth did not go to the amusement park from 9pm to 10pm.
The amusement park was closed after 10pm, so Elizabeth did not go to the amusement park after 10pm.
Im summary, the only time when Elizabeth could have gone to the amusement park was 7am to 1pm.
From the options, we can see that 7am to 1pm is the only option that fits this time frame.
So the answer is (A).""",
        "answer": "A"
    },
    {
        "problem": """Today, Tiffany went to the beach. Between what times could they have gone?
We know that:
Tiffany woke up at 5am.
Betty saw Tiffany getting a coffee at the cafe from 5am to 6am.
Jessica saw Tiffany working at the office from 6am to 9am.
John saw Tiffany stretching at a yoga studio from 9am to 12pm.
Sean saw Tiffany sitting on a rooftop from 12pm to 2pm.
Sarah saw Tiffany playing tennis at the tennis court from 2pm to 3pm.
The beach was closed after 4pm.
Between what times could Tiffany have gone to the beach?
Options:
(A) 9am to 12pm
(B) 12pm to 2pm
(C) 5am to 6am
(D) 3pm to 4pm""",
        "solution": """Tiffany wake up at 5am, so she did not go to the beach before 5am.
Tiffany wake up at 5am Betty, and saw Tiffany getting a coffee at the cafe from 5am to 6am, so Tiffany did not go to the beach from 5am to 6am.
Jessica saw Tiffany working at the office from 6am to 9am, so Tiffany did not go to the beach from 6am to 9am.
John saw Tiffany stretching at a yoga studio from 9am to 12pm, so Tiffany did not go to the beach from 9am to 12pm.
Sean saw Tiffany sitting on a rooftop from 12pm to 2pm, so Tiffany did not go to the beach from 12pm to 2pm.
Sarah saw Tiffany playing tennis at the tennis court from 2pm to 3pm, so Tiffany did not go to the beach from 2pm to 3pm.
There is no information about Tiffany's activities from 3pm to 4pm, so Tiffany could have gone to the beach from 3pm to 4pm.
The beach was closed after 4pm, so Tiffany did not go to the beach after 4pm.
Im summary, the only time when Tiffany could have gone to the beach was 3pm to 4pm.
From the options, we can see that 3pm to 4pm is the only option that fits this time frame.
So the answer is (D).""",
        "answer": "D"
    }
]


# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/formal_fallacies.txt
bbh_formal_fallacies: list[FewShotPrompt] = [
    {
        "problem": """"It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: To begin with, Lesley is a close friend of Fernando. Moreover, being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy. It follows that Lesley is a great-grandfather of Leroy."
Is the argument, given the explicitly stated premises, deductively valid or invalid?
Options:
- valid
- invalid""",
        "solution": """(1) Lesley is a close friend of Fernando: Lesley = friend(Fernando).
(2) Being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy: If X = friend(Fernando) OR SCHOOLMATE(Lowell), then X = great-grandfather(Leroy).
Hypothesis: Does it follow that Lesley is a great-grandfather of Leroy: Lesley = great-grandfather(Leroy)?
Let’s see whether the Hypothesis can be deduced from the arguments (1) and (2) by logical reasoning?
By (1), we have Lesley = friend(Fernando). By (2), we have if Lesley = friend(Fernando), then Lesley = great-grandfather(Leroy).
So, it is true that Lesley is a great-grandfather of Leroy.
So the answer is valid.""",
        "answer": "valid"
    },
    {
        "problem": """"It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: Whoever is not a great-grandfather of Clyde is a stepbrother of Brian. Being an ancestor of Dana is sufficient for not being a great-grandfather of Clyde. We may conclude: Everyone who is an ancestor of Dana is a stepbrother of Brian, too."
Is the argument, given the explicitly stated premises, deductively valid or invalid?
Options:
- valid
- invalid""",
        "solution": """(1) Whoever is not a great-grandfather of Clyde is a stepbrother of Brian: If X = NOT (great-grandfather(Clyde)), then X = stepbrother(Brian).
(2): Being an ancestor of Dana is sufficient for not being a great-grandfather of Clyde: If X = ancestor(Dana), X = NOT (great-grandfather(Clyde)).
Hypothesis: Does it follow that everyone who is an ancestor of Dana is a stepbrother of Brian, too: If X = ancestor(Dana), then X = stepbrother(Brian)?
Let’s see whether the Hypothesis can be deduced from the arguments (1) and (2) by logical reasoning?
By (2), we have if X = ancestor(Dana), X = NOT (great-grandfather(Clyde)).
Furthermore, by (1), we have if X = NOT (great-grandfather(Clyde)), then X = stepbrother(Brian).
By the transitive relation rule in first-order logic, we then have: if X = ancestor(Dana), then X = stepbrother(Brian).
So, it is true that everyone who is an ancestor of Dana is a stepbrother of Brian.
So the answer is valid.""",
        "answer": "valid"
    },
    {
        "problem": """"It is not always easy to grasp who is consuming which products. The following argument pertains to this question: Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both. No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and, in the same time, a loyal buyer of Caress soap. It follows that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap."
Is the argument, given the explicitly stated premises, deductively valid or invalid?
Options:
- valid
- invalid""",
        "solution": """(1) Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both: If X = infrequent-user(Paul Mitchell), then X = rare-consumer(Nioxin) OR X = loyal-buyer(Caress).
(2): No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and a loyal buyer of Caress soap at the same time. If X = regular-consumer(Lush), then X = NOT (rare-consumer(Nioxin) AND loyal-buyer(Caress)).
Hypothesis: Does it follow that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap: If X = infrequent-user(Paul Mitchell), then X = NOT (regular-consumer(Lush))?
Let’s see whether the Hypothesis can be deduced from the arguments (1) and (2) by logical reasoning?
By (1), we have if X = infrequent-user(Paul Mitchell), then X = rare-consumer(Nioxin) OR X = loyal-buyer(Caress). We need to consider both cases separately:
The case X = rare-consumer(Nioxin) does not appear in (2).
The case X = loyal-buyer(Caress) does not appear in (2), either.
So, from (1) and (2), we cannot necessarily deduce the Hypothesis.
So the answer is invalid.""",
        "answer": "invalid"
    }
]


# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/logical_deduction_three_objects.txt
# reasoning steps are updated by the fover authors
bbh_logical_deduction_three_objects: list[FewShotPrompt] = [
    {
        "problem": """"The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. In a golf tournament, there were three golfers: Amy, Eli, and Eve. Eve finished above Amy. Eli finished below Amy.
Options:
(A) Amy finished last
(B) Eli finished last
(C) Eve finished last""",
        "solution": """The problem says Eve finished above Amy.
The problem also says Eli finished below Amy.
Combining these two statements, we get the following ordering: "(above) Eve Amy Eli (below)". 
According to this ordering, the person who finished last (the one at the bottom of this list) is Eli.
So the answer is (B).""",
        "answer": "B"
    },
    {
        "problem": """"The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: a white book, a green book, and an orange book. The green book is to the right of the white book. The orange book is the rightmost.
Options:
(A) The white book is the leftmost
(B) The green book is the leftmost
(C) The orange book is the leftmost""",
        "solution": """The problem says the green book is to the right of the white book.
The problem also says the orange book is the rightmost.
Combining these two statements, we get the following ordering: "(left) white green orange (right)".
According to this ordering, the leftmost book is the white book.
So the answer is (A).""",
        "answer": "A"
    },
    {
        "problem": """"The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: a red book, a gray book, and a white book. The white book is to the left of the gray book. The red book is the second from the left.
Options:
(A) The red book is the leftmost
(B) The gray book is the leftmost
(C) The white book is the leftmost""",
        "solution": """The problem says the white book is to the left of the gray book.
The problem also says the red book is the second from the left.
Combining these two statements, we can deduce the following: "(left) white red gray (right)".
According to this ordering, the leftmost book is the white book.
So the answer is (C).""",
        "answer": "C"
    }
]


# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/boolean_expressions.txt
# reasoning steps are updated by the fover authors
bbh_boolean_expressions: list[FewShotPrompt] = [
    {
        "problem": """"not ( ( not not True ) ) is""",
        "solution": """We first evaluate the inner expression "not not True". "not not True = not (not True) = not False = True".
Plugging this into the outer expression, we get: "not ( ( not not True ) ) = not ( ( True ) )".
Let's evaluate the whole expression: "not ( ( True ) ) = not True = False".
So the answer is False.""",
        "answer": "False"
    },
    {
        "problem": """"True and False and not True and True is""",
        "solution": """We first evaluate the first part of the expression: "True and False = False".
Plugging this into the expression, we get: "False and not True and True".
Now we evaluate the first part of the above expression: "False and not True = False and False = False".
Plugging this into the expression, we get: "False and True".
Now we evaluate the whole expression: "False and True = False".
So the answer is False.""",
        "answer": "False"
    },
    {
        "problem": """"not not ( not ( False ) ) is""",
        "solution": """We first evaluate the inner expression "not ( False )". "not ( False ) = True".
Plugging this into the expression, we get: "not not ( True )".
Now we evaluate the whole expression: "not not ( True ) = not (not True) = not False = True".
So the answer is True.""",
        "answer": "True"
    }
]
