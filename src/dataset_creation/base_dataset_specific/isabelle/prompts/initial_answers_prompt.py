""" This file contains the prompts for the initial outputs generation. We then use the verifier to verify the initial outputs to create datasets for training verifiers. """

import random

from src.dataset_creation.prompts import get_assistant_message, get_user_message 


###
# First-order logic: MetaMathQA

isabelle_prompt_conversation: list[dict] = []

# selected from training data
fol_few_shot_demonstrations = [
    {
        "question": "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
        "solution": """The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.
In this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.
So the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}$.
Therefore, Gracie and Joe's points are $\\boxed{\\sqrt{5}}$ units apart.
The answer is: \\sqrt{5}"""
    },
    {
        "question": "What is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?",
        "solution": """Each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80.
So the total cost for each player is $25 + $15.20 + $6.80 = $47.
Since there are sixteen players on the football team, the total cost for all of them is 16 * $47 = $752.
The answer is: 752"""
    }
]


isabelle_user_template = """{question}"""
isabelle_assistant_template = """{solution}"""


def get_isabelle_few_shot_examples(model_name: str) -> list[dict]:
    # convert the few-shot examples to the prompt format
    isabelle_prompt_conversation = []
    for example in fol_few_shot_demonstrations:
        # input user message
        isabelle_prompt_conversation.append(
            get_user_message(example["question"])
        )

        # expected assistant message
        isabelle_prompt_conversation.append(
            get_assistant_message(
                example["solution"],
                model_name=model_name
            )
        )
    
    return isabelle_prompt_conversation


###
# Generate prompt for initial generation
def get_isabelle_initial_generation_prompt(model_name: str, instance: dict[str, str]) -> list[dict]:
    """ Get the initial generation prompt (conversation with few-shot examples) for the given instance. """
    
    # few-shot examples
    fld_prompt_conversation = get_isabelle_few_shot_examples(model_name)
    
    # append the user message for the given instance
    fld_prompt_conversation.append(get_user_message(instance["question"]))
    
    return fld_prompt_conversation
