# We created the few-shot demonstrations by ourselves

# The task instruction is based on the following source and modified by us
# https://github.com/hitachi-nlp/lm-evaluation-harness/blob/ba7b684dbdcfaad32631d08d378b641e69c01add/lm_eval/tasks/folio/folio.yaml

from src.downstream_evaluation.prompts.typing import FewShotPrompt


def get_folio_instruction(premise: str, hypothesis: str) -> str:
    return f"""Premise: {premise}
Conclusion: {hypothesis}
Question: do the premises derive the conclusion? Provide reasoning and answer with either \"True\", \"False\" or \"Uncertain\"."""


folio_fewshot_prompt: list[FewShotPrompt] = [
    {
        # story_id: 8
        # example_id: 21
        "problem": get_folio_instruction(
            premise="""Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.""",
            hypothesis="A Czech published a book in 1946.",
        ),
        "solution": """Premise says, "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music." Therefore, Miroslav Venhoda was a Czech.
We know Miroslav Venhoda was a Czech, and Premise says, "Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant." Therefore, a Czech published a book in 1946.
We know a Czech published a book in 1946, and this is consistent with the Conclusion, "A Czech published a book in 1946."
Therefore, the final answer is True.""",
        "answer": "True"
    },
    {
        # story_id: 463
        # example_id: 1,337
        "problem": get_folio_instruction(
            premise="""All eels are fish.
No fish are plants.
Everything displayed in the collection is either a plant or an animal.
All multicellular animals are not bacteria.
All animals displayed in the collection are multicellular.
A sea eel is displayed in the collection.
The sea eel is an eel or an animal or not a plant.""",
            hypothesis="The sea eel is bacteria.",
        ),
        "solution": """Premise says, "All eels are fish" and "No fish are plants." Therefore, sea eels are not plants.
Premise says, "A sea eel is displayed in the collection" and "Everything displayed in the collection is either a plant or an animal." Therefore, sea eels are plants or animals.
We know sea eels are not plants and are either plants or animals. Therefore, sea eels are animals.
We know sea eels are animals, and Premise says, "A sea eel is displayed in the collection." Therefore, see eels are animals displayed in the collection.
we know sea eels are animals displayed in the collection and Premise says, "All animals displayed in the collection are multicellular." Therefore, sea eels are multicellular.
we know sea eels are multicellular and Premise says, "All multicellular animals are not bacteria." Therefore, sea eels are not bacteria.
We know sea eels are not bacteria, and it contradicts the Conclusion, "The sea eel is bacteria."
Therefore, the final answer is False.""",
        "answer": "False"
    },
    {
        # story_id: 226
        # example_id: 637
        "problem": get_folio_instruction(
            premise="""William Dickinson was a British politician who sat in the House of Commons William Dickinson attended Westminster school for high school and then the University of Edinburgh.
The University of Edinburgh is a university located in the United Kingdom.
William Dickinson supported the Portland Whigs.
People who supported the Portland Whigs did not get a seat in the Parliament.""",
            hypothesis="William Dickinson went to schools located in the United Kingdom for both high school and university.",
        ),
        "solution": """Premise says, "William Dickinson was a British politician who sat in the House of Commons William Dickinson attended Westminster school for high school and then the University of Edinburgh." Therefore, William Dickinson attended Westminster school for high school and then the University of Edinburgh.
We know William Dickinson attended the University of Edinburgh, and Premise says, "The University of Edinburgh is a university located in the United Kingdom." Therefore, William Dickinson went to a university located in the United Kingdom.
However, Premise does not say anything about Westminster school being located in the United Kingdom.
Therefore, the Conclusion, "William Dickinson went to schools located in the United Kingdom for both high school and university," is Uncertain.
Therefore, the final answer is Uncertain.""",
        "answer": "Uncertain"
    }
]
