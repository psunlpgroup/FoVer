from src.dataset_creation.prompts import get_assistant_message, get_user_message
from src.downstream_evaluation.prompts.dataset_prompts.gsm8k \
    import gsm8k_fewshot_prompt, gsm8k_fewshot_prompt_for_training_data
from src.downstream_evaluation.prompts.dataset_prompts.math \
    import math_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.aqua \
    import aqua_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.orca_math \
    import orca_math_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.folio \
    import folio_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.logicnli \
    import logicnli_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.anli \
    import anli_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.hans \
    import hans_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.mmlu_pro_nomath \
    import mmlu_pro_nomath_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.aime \
    import aime_fewshot_prompt
from src.downstream_evaluation.prompts.dataset_prompts.bbh \
    import bbh_word_sorting_prompt, \
        bbh_tracking_shuffled_objects_three_objects, \
        bbh_temporal_sequences, bbh_formal_fallacies, \
        bbh_logical_deduction_three_objects, \
        bbh_boolean_expressions

downstream_evaluation_datasets_prompts_dict = {
    "gsm8k": gsm8k_fewshot_prompt,
    "gsm8k_for_training_data": gsm8k_fewshot_prompt_for_training_data,
    "math": math_fewshot_prompt,
    "aqua": aqua_fewshot_prompt,
    "orca_math": orca_math_fewshot_prompt,
    "folio": folio_fewshot_prompt,
    "logicnli": logicnli_fewshot_prompt,
    "anli": anli_fewshot_prompt,
    "hans": hans_fewshot_prompt,
    "mmlu_pro_nomath": mmlu_pro_nomath_fewshot_prompt,
    "aime": aime_fewshot_prompt,
    "bbh_formal_fallacies": bbh_formal_fallacies,
    "bbh_logical_deduction_three_objects": bbh_logical_deduction_three_objects,
    "bbh_temporal_sequences": bbh_temporal_sequences,
    "bbh_tracking_shuffled_objects_three_objects": \
        bbh_tracking_shuffled_objects_three_objects,
    "bbh_word_sorting": bbh_word_sorting_prompt,
    # "bbh_navigate": bbh_navigate,
    "bbh_boolean_expressions": bbh_boolean_expressions,
}


new_question_instruction = "Your response to the following question should follow the format (e.g., structure, style, line breaks) of responses in previous examples. Specifiaclly, you should provide reasoning where steps are separated by line breaks."""

def make_few_shot_example_user_message(
    problem: str, model_name: str, dataset_name: str) -> str:
    
    if "Llama" in model_name:
        # llama is already good at following few-shot examples
        return problem
    else:
        instruction = new_question_instruction
        if dataset_name == "bbh_word_sorting":
            instruction += " The answer should be a list of words separated by a space."

        return instruction + "\n\n" + problem


def get_fewshot_prompt_for_initial_generation_of_sample_and_rank(
    dataset_name: str, model_name: str, new_question: str) -> list[dict]:
    """ Get few-shot prompt for initial generation of downstream evaluation in chat format. """
    
    # we use the few-shot examples for gsm8k for metamathqa_gsm8k
    # because the format is the same
    if dataset_name == "metamathqa_gsm8k":
        dataset_name = "gsm8k_for_training_data"
    
    few_shot_examples = downstream_evaluation_datasets_prompts_dict[dataset_name]

    # few-shot examples
    prompt_in_chat_format: list[dict] = []
    for example in few_shot_examples:
        prompt_in_chat_format.append(
            get_user_message(
                make_few_shot_example_user_message(
                    problem=example["problem"],
                    model_name=model_name,
                    dataset_name=dataset_name,
                )
            )
        )
        prompt_in_chat_format.append(get_assistant_message(example["solution"], model_name))
    
    # new input
    prompt_in_chat_format.append(
        get_user_message(
            make_few_shot_example_user_message(
                problem=new_question,
                model_name=model_name,
                dataset_name=dataset_name,
            )
        )
    )
    
    return prompt_in_chat_format


answer_extraction_template = """

Your task: extract the final answer from the above solution and write it in the box $\\boxed{}$. Don't generate anything other than the final answer."""

answer_extraction_template_with_options = """

Your task: extract the final answer ({options}) from the above solution and write it in the box $\\boxed{{}}$. Don't generate anything other than the final answer."""


def get_answer_extraction_template(dataset_name: str) -> str:
    if dataset_name in ["bbh_tracking_shuffled_objects_three_objects",
                        "bbh_logical_deduction_three_objects"]:
        return answer_extraction_template_with_options.format(
            options="'A', 'B', or 'C'"
        )
    elif dataset_name == "bbh_temporal_sequences":
        return answer_extraction_template_with_options.format(
            options="'A', 'B', 'C', or 'D'"
        )
    elif dataset_name == "bbh_navigage":
        return answer_extraction_template_with_options.format(
            options="'Yes' or 'No'"
        )
    elif dataset_name == "bbh_boolean_expressions":
        return answer_extraction_template_with_options.format(
            options="'True' or 'False'"
        )
    elif dataset_name == "aqua":
        return answer_extraction_template_with_options.format(
            options="'A', 'B', 'C', 'D', or 'E'"
        )
    elif dataset_name == "bbh_formal_fallacies":
        return answer_extraction_template_with_options.format(
            options="'valid' or 'invalid'"
        )
    elif dataset_name == "folio":
        return answer_extraction_template_with_options.format(
            options="'True', 'False', or 'Uncertain'"
        )
    elif dataset_name in ["logicnli", "anli"]:
        return answer_extraction_template_with_options.format(
            options="'entailment', 'contradiction', or 'neutral'"
        )
    elif dataset_name == "hans":
        return answer_extraction_template_with_options.format(
            options="'entailment' or 'non-entailment'"
        )
    elif dataset_name == "mmlu_pro_nomath":
        return answer_extraction_template_with_options.format(
            options="'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', ..."
        )
    else:
        # no options (e.g., GSM8K, Math, AIME, BBH word sorting)
        return answer_extraction_template


def get_fewshot_prompt_for_answer_extraction_of_sample_and_rank(
    dataset_name: str, model_name: str, new_solution: str) -> list[dict]:
    """ Get few-shot prompt for answer extraction in chat format. """
    
    # we use the few-shot examples for gsm8k for metamathqa_gsm8k
    # because the format is the same
    if dataset_name == "metamathqa_gsm8k":
        dataset_name = "gsm8k_for_training_data"
    
    few_shot_examples = downstream_evaluation_datasets_prompts_dict[dataset_name]
    
    few_shot_in_chat_format: list[dict] = []
    for example in few_shot_examples:
        # solution and instruction
        few_shot_in_chat_format.append(
            get_user_message(
                example["solution"]
                + get_answer_extraction_template(dataset_name)
            )
        )
        
        # final answer
        answer = example["answer"]
        answer_with_template = f"$\\boxed{{{answer}}}$"
        few_shot_in_chat_format.append(
            get_assistant_message(answer_with_template, model_name)
        )
    
    # new input
    few_shot_in_chat_format.append(
        get_user_message(
            new_solution + get_answer_extraction_template(dataset_name)
        )
    )
    
    return few_shot_in_chat_format
