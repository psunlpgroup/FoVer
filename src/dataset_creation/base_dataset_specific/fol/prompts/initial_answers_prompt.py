""" This file contains the prompts for the initial outputs generation. We then use the verifier to verify the initial outputs to create datasets for training verifiers. """

import random

from src.dataset_creation.prompts import get_assistant_message, get_user_message 


###
# First-order logic: FLDx2

fld_prompt_conversation: list[dict] = []

# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/fld/fld_logical_formula_default.yaml
# (slightly updated)
fol_symbol_problem_definition = "Based on the provided facts ($context$), either prove or disprove the hypothesis or state that it is unknown. " \
    "The facts and the hypothesis are written in logical formulas as follows: capital letters such as \"{A}\", \"{B}\", \"{AB}\" are predicates, " \
    "small letters such as \"{a}\", \"{b}\", \"{ab}\" are constants, \"&\" is logical conjunction, \"v\" is logical disjunction, " \
    "\"¬\" is negation, \"->\" is implication, \"(x)\" is \"for all x\", and \"(Ex)\" is \"for some x\"."

# added by the FoVer author
fol_problem_detailed_instruction = "Don't anything other than the proof and proof_label.\n" \
    "Don't generate the proof by contradiction. If the hypothesis is disproved, provide a direct counterexample. " \
    "If the proof_label is PROVED, your proof should end with the 'hypothesis'. If the proof_label is DISPROVED, your proof should end with '¬hypothesis'."

# few shot examples
# these examples are randomly selected from training data
# we use these few-shot examples for generating initial responses for training data as well,
# but it is ok to use these few-shot examples because we are not evaluating the quality of initial generation on training data
fol_few_shot_demonstrations_for_each_label_dict = {
    "PROVED": [
    {
        "hypothesis_formula": "{D}",
        "facts_formula": """fact1: {B} -> (¬{D} v ¬{A})
fact2: (¬{BG} & ¬{IK}) -> {HK}
fact3: {B} -> (¬{A} & ¬{D})
fact4: {M} -> {L}
fact5: ¬{HL}
fact6: ¬{E} -> ({B} & {C})
fact7: ({F} & {G}) -> ¬{E}
fact8: {C} -> {D}
fact9: ¬(¬{A} & ¬{FI}) -> {FI}
fact10: {G} -> (¬{E} & ¬{F})
fact11: {K} -> (¬{J} & ¬{I})
fact12: ¬(¬{G} & ¬{K}) -> {G}
fact13: ¬{I} -> ({F} & {H})
fact14: (¬{A} & ¬{B}) -> {C}
fact15: ¬{I} -> ({H} & {G})
fact16: {FU}
fact17: ¬{A}
fact18: {L} -> {K}
fact19: ¬{B}""",
        "proof": """fact17 & fact19 -> int1: (¬{A} & ¬{B});
int1 & fact14 -> int2: {C};
int2 & fact8 -> hypothesis;""",
        "proof_label": "PROVED"
    },
    {
        "hypothesis_formula": "¬({C}{b} & {D}{b})",
        "facts_formula": """fact1: {B}{b}
fact2: (x): ({A}x & {B}x) -> ¬{C}x
fact3: ¬{C}{a} -> ¬({C}{b} & {D}{b})
fact4: {A}{a}
fact5: ¬{D}{b} -> (¬{D}{a} & ¬{A}{a})
fact6: {B}{a}""",
        "proof": """fact4 & fact6 -> int1: ({A}{a} & {B}{a});
fact2 -> int2: ({A}{a} & {B}{a}) -> ¬{C}{a};
int1 & int2 -> int3: ¬{C}{a};
int3 & fact3 -> hypothesis;""",
        "proof_label": "PROVED"
    },
    {
        "hypothesis_formula": "({D} v ¬{C})",
        "facts_formula": """fact1: ({GP} v {ED})
fact2: ({E} v ¬{FI})
fact3: ({D} v {C})
fact4: {A}
fact5: {EE}
fact6: ({BP} & {CP})
fact7: {HS}
fact8: {B}
fact9: ({R} & {DT}) -> ¬{GH}
fact10: ¬{BD}
fact11: ({A} & {B}) -> ¬{C}
fact12: ({CJ} v {GL})
fact13: {FB}
fact14: ¬{DP}
fact15: {CR}
fact16: ¬{A} -> ¬({D} v ¬{C})
fact17: ¬{S}""",
        "proof": """fact4 & fact8 -> int1: ({A} & {B});
int1 & fact11 -> int2: ¬{C};
int2 -> hypothesis;""",
        "proof_label": "PROVED"
    }
    ],
    "DISPROVED": [
    {
        "hypothesis_formula": "¬({C} & {D})",
        "facts_formula": """fact1: {A}
fact2: {B} -> {C}
fact3: ¬{F}
fact4: ¬{GU}
fact5: {A} -> {B}
fact6: {JI} -> {DA}
fact7: ¬{HE}
fact8: {CD}
fact9: ¬{F} -> ({D} & ¬{E})""",
        "proof": """fact5 & fact1 -> int1: {B};
int1 & fact2 -> int2: {C};
fact9 & fact3 -> int3: ({D} & ¬{E});
int3 -> int4: {D};
int2 & int4 -> ¬hypothesis;""",
        "proof_label": "DISPROVED"
    },
    {
        "hypothesis_formula": "¬({AA}{aa} & ¬{AB}{aa})",
        "facts_formula": """fact1: (x): {A}x -> ¬(¬{F}x & ¬{G}x)
fact2: (x): {D}x -> {A}x
fact3: {A}{a} -> ¬({AA}{aa} & ¬{AB}{aa})
fact4: ¬({C}{aa} & ¬{B}{aa})
fact5: (x): ¬(¬{F}x & ¬{G}x) -> ¬{GH}x
fact6: (x): ¬{A}x -> ({AA}x & ¬{AB}x)
fact7: ¬({C}{aa} & ¬{B}{aa}) -> ¬{A}{aa}
fact8: (x): {E}x -> {B}x
fact9: (x): ¬{C}x -> {E}x
fact10: (x): ¬({CB}x & ¬{AP}x) -> ¬{B}x
fact11: (x): (¬{C}x v {D}x)""",
        "proof": """fact6 -> int1: ¬{A}{aa} -> ({AA}{aa} & ¬{AB}{aa});
fact4 & fact7 -> int2: ¬{A}{aa};
int1 & int2 -> ¬hypothesis;""",
        "proof_label": "DISPROVED"
    },
    {
        "hypothesis_formula": "¬{C}{b}",
        "facts_formula": """fact1: (x): {E}x -> {B}x
fact2: ¬{D}{a} -> ¬{A}{b}
fact3: ¬{K}{f} -> ({J}{c} & {H}{c})
fact4: {I}{e} -> {E}{d}
fact5: ({B}{a} & {A}{a}) -> ¬{C}{b}
fact6: ¬{F}{a} -> (¬{E}{a} & ¬{D}{a})
fact7: (x): ¬({E}x & {G}x) -> ¬{E}x
fact8: (x): ¬{F}x
fact9: (x): ¬{F}x -> ¬({E}x & {G}x)
fact10: ¬(¬{K}{g} & {M}{g}) -> ¬{K}{f}
fact11: (x): {H}x -> ¬{F}x
fact12: ¬({G}{c} & ¬{F}{c}) -> ¬{F}{a}
fact13: (x): ¬{F}x -> ({D}x v {G}x)
fact14: ¬(¬{K}{g} & {M}{g})
fact15: ¬{A}{a}
fact16: {D}{c} -> {D}{a}
fact17: {E}{d} -> {E}{a}
fact18: (x): ¬{B}x -> (¬{B}x & {D}x)
fact19: (x): ({D}x & {A}x) -> {B}x
fact20: ({B}{b} & {D}{b}) -> {C}{b}
fact21: (x): (¬{B}x & {D}x) -> {C}x
fact22: (x): ({B}x & {D}x) -> {C}x
fact23: ¬{E}{b} -> ¬{D}{a}
fact24: ¬{A}{a} -> ¬{B}{b}""",
        "proof": """fact24 & fact15 -> int1: ¬{B}{b};
fact18 -> int2: ¬{B}{b} -> (¬{B}{b} & {D}{b});
int1 & int2 -> int3: (¬{B}{b} & {D}{b});
fact21 -> int4: (¬{B}{b} & {D}{b}) -> {C}{b};
int3 & int4 -> ¬hypothesis;""",
        "proof_label": "DISPROVED"
    }
    ],
    "UNKNOWN": [
    {
        "hypothesis_formula": "({AB}{c} & ¬{B}{c})",
        "facts_formula": """fact1: (x): ¬{D}x -> ({A}x & {C}x)
fact2: ¬({B}{a} & ¬{A}{a})
fact3: ¬{AB}{b} -> ¬{C}{d}
fact4: ¬{C}{a} -> ¬{A}{d}
fact5: ¬{A}{a}
fact6: ({AB}{c} & ¬{B}{c}) -> {C}{d}
fact7: ¬({F}{b} v {D}{b}) -> ¬{D}{a}
fact8: {A}{a} -> ({AB}{c} & ¬{B}{c})
fact9: ({AB}{c} & ¬{AA}{c}) -> {B}{b}
fact10: ({AB}{c} & ¬{B}{c}) -> {C}{b}""",
        "proof": """fact6 -> int1: ¬{C}{d} -> ¬({AB}{c} & ¬{B}{c});
fact3 & int1 -> int2: ¬{AB}{b} -> ¬({AB}{c} & ¬{B}{c});""",
        "proof_label": "UNKNOWN"
    },
    {
        "hypothesis_formula": "¬({C} & {B})",
        "facts_formula": """fact1: ({A} & {B})
fact2: ¬{F} -> ({D} & {E})
fact3: {A}
fact4: ¬{F} -> ¬({D} & {E})
fact5: {A} -> {HU}
fact6: ({CQ} & {IG})
fact7: ¬{C} -> ({B} & {A})
fact8: ¬{I} -> ¬(¬{H} v ¬{G})
fact9: ¬{A} -> ¬({C} & {B})
fact10: {D}
fact11: ({CE} & {BT})
fact12: ({AG} & {IO})
fact13: ¬(¬{H} v ¬{G}) -> ¬{F}
fact14: ¬({D} & {E}) -> ¬{C}
fact15: {IJ}""",
        "proof": "fact1 -> int1: {B};",
        "proof_label": "UNKNOWN"
    },
    {
        "hypothesis_formula": "{D}{b}",
        "facts_formula": """fact1: (¬{C}{a} & {B}{a}) -> {C}{df}
fact2: {G}{b} -> ¬{A}{a}
fact3: ¬{E}{c} -> ¬(¬{D}{b} & {A}{b})
fact4: ({D}{a} & {CG}{a})
fact5: (x): {FH}x -> {FE}x
fact6: ({A}{a} & {B}{a})
fact7: {C}{b} -> {B}{b}
fact8: {HR}{b}
fact9: (x): {CA}x -> {IM}x
fact10: (x): ¬(¬{C}x & {E}x) -> ¬{B}{c}\
fact11: {A}{cn}
fact12: (x): {F}x -> ¬(¬{C}x & {E}x)
fact13: {D}{b} -> {B}{b}
fact14: (x): ¬{B}x -> ({C}x & ¬{A}x)
fact15: {GQ}{hp} -> {CL}{hp}
fact16: ¬{B}{c} -> ¬({A}{a} & ¬{D}{a})
fact17: (x): {B}x -> {A}x
fact18: {D}{b} -> ({G}{b} v ¬{F}{b})
fact19: {B}{a} -> {C}{b}
fact20: {DN}{b}""",
        "proof": """fact6 -> int1: {B}{a};
int1 & fact19 -> int2: {C}{b};""",
        "proof_label": "UNKNOWN"
    }
    ]
}


fol_user_template = """$hypothesis$: {hypothesis_formula}

$context$:
{facts_formula}"""

fol_assistant_template = """$proof$:
{proof}

$proof_label$: {proof_label}"""


def get_fld_few_shot_examples(model_name: str, seed: str, num_examples: int=6) -> list[dict]:
    if num_examples % 3 != 0:
        # we select the same number of examples for each label\
        # (PROVED, DISPROVED, UNKNOWN)
        raise ValueError("Number of examples should be a multiple of 3.")
    num_example_for_each_label = num_examples // 3
    
    # select few-shot examples for each label
    fol_few_shot_demonstrations_list = []
    for label, examples in fol_few_shot_demonstrations_for_each_label_dict.items():
        selected_examples = random.Random(f"fld_{seed}_{label}").sample(
            examples, num_example_for_each_label
        )
        fol_few_shot_demonstrations_list.extend(selected_examples)
    
    # shuffle the examples
    fol_few_shot_demonstrations_list = random.Random(
        f"fld_{seed}_shuffle").sample(
            fol_few_shot_demonstrations_list, num_examples
        )
    
    # convert the few-shot examples to the prompt format
    fld_prompt_conversation = []
    for idx, example in enumerate(fol_few_shot_demonstrations_list):
        # input user message
        user_message = ""
        if idx == 0:
            # for the first example, add the problem definition
            user_message += fol_symbol_problem_definition + "\n\n" + fol_problem_detailed_instruction + "\n\n"
        
        user_message += fol_user_template.format(
            hypothesis_formula=example["hypothesis_formula"],
            facts_formula=example["facts_formula"]
        )
        
        fld_prompt_conversation.append(
            get_user_message(user_message)
        )

        # expected assistant message
        fld_prompt_conversation.append(
            get_assistant_message(
                fol_assistant_template.format(
                    proof=example["proof"],
                    proof_label=example["proof_label"]
                ),
                model_name=model_name
            )
        )
    
    return fld_prompt_conversation


###
# Generate prompt for initial generation
def get_fld_initial_generation_prompt(model_name: str, instance: dict[str, str], seed: int, num_examples: int=6) -> list[dict]:
    """ Get the initial generation prompt (conversation with few-shot examples) for the given instance. """
    
    # user message for the new instance
    hypothesis_formula = instance["hypothesis_formula"]
    facts_formula = instance["facts_formula"]
    user_message = fol_user_template.format(hypothesis_formula=hypothesis_formula, facts_formula=facts_formula)
    
    # append to the conversation with few-shot examples
    fld_prompt_conversation = get_fld_few_shot_examples(model_name, seed=seed, num_examples=num_examples)
    fld_prompt_conversation.append(get_user_message(user_message))
    
    return fld_prompt_conversation
