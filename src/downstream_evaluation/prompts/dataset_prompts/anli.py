from src.downstream_evaluation.prompts.typing import FewShotPrompt


# few shot examples are from training set of anli
# https://huggingface.co/datasets/facebook/anli


def get_anli_instruction(premise: str, hypothesis: str) -> str:
    return f"""Premise: {premise}
Hypothesis: {hypothesis}
Question: do the premises derive the hypothesis? Provide reasoning and answer with either \"entailment\", \"contradiction\" or \"neutral\"."""


anli_fewshot_prompt: list[FewShotPrompt] = [
    {
        # d7331d13-1b3b-4423-a3b2-8ac71951b64f
        "problem": get_anli_instruction(
            premise="""Tallahassee Mayor and Democratic gubernatorial candidate Andrew Gillum, speaks during a briefing on Hurricane Michael in Tallahassee, Fla., Wednesday Oct. 10, 2018. In the days before Hurricane Michael made landfall, Gillum blanketed the national airwaves, sounding much like the man he wants to replace, Gov. Rick Scott, who constantly urged residents to seek shelter and take the massive storm seriously. Gary Fineout AP Photo""",
            hypothesis="Gillum was on TV urging residents to stay out of the storm.",
        ),
        "solution": """Premise says, "Gillum blanketed the national airwaves," implying that he was on TV.
Premise says, "sounding much like the man [...] constantly urged residents to seek shelter and take the massive storm seriously," implying that Gillum was urging residents to stay out of the storm.
In summary, the premise implies that Gillum was on TV and was urging residents to stay out of the storm. Therefore, the hypothesis is supported by the premise.
Therefore, the final answer is entailment.""",
        "answer": "entailment"
    },
    {
        # eb293bfd-c891-4033-9e50-8cab39aac594
        "problem": get_anli_instruction(
            premise="""A man walks past an electronic stock board showing Japan's Nikkei 225 index and other country's index at a securities firm in Tokyo Monday, Sept. 3, 2018. Asian shares were mostly lower Monday amid worries about escalating trade friction between the U.S. and Canada, who have been unable to agree to a revamped trade deal but will continue negotiating this week. Eugene Hoshiko AP Photo""",
            hypothesis="The US Canada and Japan are close to concluding a trade deal.",
        ),
        "solution": """Premise says, "who have been unable to agree to a revamped trade deal but will continue negotiating this week." However, it does not say anything about the US and Canada being close to concluding a trade deal.
Premise does not imply that the US and Canada are close to concluding a trade deal, but it does not contradict it either.
Therefore, the final answer is neutral.""",
        "answer": "neutral"
    },
    {
        # 778cfe29-faae-4031-aeee-8233aa4f9655
        "problem": get_anli_instruction(
            premise="""Diamond Dawgs in town for the MSU World Series. A weekend of relaxed practicing, scrimmaging and home run derby fun. MSU head coach Andy Cannizaro spoke about the 2017-2018 team. "It's going to be a really interesting group this year. We have several talented young guys," Andy Cannizaro said. "Justin Foscue and Jordan Westburg had outstanding falls for us this year. Tanner Allen had a really nice fall as well as Josh Hatcher." Copyright 2017 MSNewsNow. All rights reserved.""",
            hypothesis="Andy Cannizaro was a player on the 2017-2018 MSU team.",
        ),
        "solution": """Premise says, MSU head coach Andy Cannizaro spoke about the 2017-2018 team. "It's going to be a really interesting group this year," suggesting that Andy Cannizaro was head coach of the 2017-2018 MSU team.
It implies that Andy Cannizaro was not a player on the 2017-2018 MSU team.
In summary, the premise implies that Andy Cannizaro was not a player on the 2017-2018 MSU team, and the hypothesis contradicts the premise.
Therefore, the final answer is contradiction.""",
        "answer": "contradiction"
    }
]
