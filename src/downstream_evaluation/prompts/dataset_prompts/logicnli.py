# We created the few-shot demonstrations by ourselves
# using training data https://huggingface.co/datasets/tasksource/LogicNLI/viewer/default/train

# The task instruction is based on the following source and modified by us
# https://github.com/hitachi-nlp/lm-evaluation-harness/blob/ba7b684dbdcfaad32631d08d378b641e69c01add/lm_eval/tasks/logic_nli/logic_nli.yaml

from src.downstream_evaluation.prompts.typing import FewShotPrompt


def get_logicnli_instruction(premise: str, hypothesis: str) -> str:
    return f"""Premise: {premise}
Conclusion: {hypothesis}
Question: do the premises derive the conclusion? Provide reasoning and answer with either \"entailment\", \"contradiction\" or \"neutral\"."""


logicnli_fewshot_prompt: list[FewShotPrompt] = [
    {
        "problem": get_logicnli_instruction(
            premise="""Nathalie is not blue.
Gabriel is concerned.
Nathalie is not concerned.
Baird is concerned.
Baird is serious.
Quinlan is not entire.
John is not fresh.
John is blue.
Gabriel is serious.
Arthur is serious.
Gabriel is not entire.
Nathalie is not accurate.If there is someone who is either not concerned or fresh, then John is entire.
Someone who is eithor not fresh or entire is always not serious.
If John is not serious and Quinlan is fresh, then Baird is not entire.
If Gabriel is not serious, then Quinlan is not blue and Gabriel is accurate.
If Nathalie is not blue, then Collier is entire.
If someone is fresh or he is not concerned, then he is not blue.
If there is someone who is both serious and fresh, then Baird is not blue.
If someone is concerned and serious, then he is entire, and vice versa.
Nathalie being not fresh and Baird being serious imply that Nathalie is blue.
If there is at least one people who is both not accurate and entire, then Collier is serious.
It can be concluded that Gabriel is concerned once knowing that Baird is not blue.
If someone is entire, then he is not serious, and vice versa.""",
            hypothesis="John is not entire.",
        ),
        "solution": """Premise says, "Nathalie is not concerned."
Premise says, "If there is someone who is either not concerned or fresh, then John is entire."
Therefore, Nathalie is not concerned implies John is entire.
Therefore, the Conclusion, "John is not entire" contradicts the premise.
Therefore, the answer is contradiction.""",
        "answer": "contradiction"
    },
    {
        "problem": get_logicnli_instruction(
            premise="""Jerry is not excited.
Lloyd is not excited.
Jed is not courteous.
Harris is various.
Lloyd is not timid.
Hope is various.
Lloyd is not disgusted.
Hope is excited.
Harris is timid.
Harris is not disgusted.
Jonathan is disgusted.
Jonathan is excited.If there is at least one people who is various, then Jonathan is not elderly and Vera is disgusted.
If there is at least one people who is both not excited and not timid, then Jonathan is elderly.
Someone being excited is equivalent to being various.
It can be concluded that Jerry is disgusted once knowing that Lloyd is timid and Lloyd is various.
If someone is not excited, then he is not disgusted.
If there is at least one people who is not courteous, then Harris is elderly.
If Jerry is courteous and Jerry is disgusted, then Vera is not elderly.
If there is someone who is either timid or not elderly, then Jonathan is not various.
Someone is courteous and not elderly if and only if he is not excited and not various.
Someone who is both not disgusted and courteous is always elderly.
If there is at least one people who is either not disgusted or not elderly, then Harris is not timid and Jed is not various.
Harris being various implies that Vera is elderly and Lloyd is timid.""",
            hypothesis="Harris is excited.",
        ),
        "solution": """Premise says, "Harris is various."
Premise says, "Someone being excited is equivalent to being various."
Therefore, Harris being various implies Harris is excited.
Therefore, the Conclusion, "Harris is excited" is consistent with the premise.
Therefore, the answer is entailment.""",
        "answer": "entailment"
    },
    {
        "problem": get_logicnli_instruction(
            premise="""Rick is yellow.
Luna is not rational.
Luna is not open.
Aubrey is open.
Herbert is sane.
Joe is sane.
Joe is troubled.
Luna is not troubled.
Herbert is unhappy.
Angus is troubled.
Joe is open.
Luna is yellow.It can be concluded that Aubrey is not sane and Hope is not rational once knowing that Herbert is troubled.
If all people are sane or not open, then Joe is not rational.
Aubrey being not troubled or Aubrey being not rational implies that Herbert is not unhappy.
If Hope is not yellow, then Herbert is troubled and Aubrey is rational, and vice versa.
If there is at least one people who is both open and rational, then Hope is not troubled.
Someone is rational and not yellow if and only if he is open.
If there is someone who is not unhappy, then Angus is not troubled and Angus is not yellow.
Someone being both not sane and not rational is equivalent to being yellow and not troubled.
If Rick is rational or Rick is unhappy, then Herbert is troubled.
Someone who is not troubled is always sane.
Someone who is not rational or not unhappy is always open and not yellow.
If there is someone who is either rational or sane, then Luna is not open.""",
            hypothesis="Herbert is not yellow.",
        ),
        "solution": """Premise says, "Someone being both not sane and not rational is equivalent to being yellow and not troubled."
However, Premise does not say anything about Herbert being rational or not.
Premise says, "Someone who is not rational or not unhappy is always open and not yellow."
However, Premise says, "Herbert is unhappy."
Therefore, from Premise, we can not conclude that Herbert is yellow or not.
Therefore, the answer is neutral.""",
        "answer": "neutral"
    },
]
