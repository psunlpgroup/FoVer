""" This file contains the prompts for the initial outputs generation. We then use the verifier to verify the initial outputs to create datasets for training verifiers. """

from src.dataset_creation.base_dataset_specific.fol.prompts.initial_answers_prompt import get_fld_initial_generation_prompt
from src.dataset_creation.base_dataset_specific.isabelle.prompts.initial_answers_prompt import get_isabelle_initial_generation_prompt


###
# Generate prompt for initial generation

def get_initial_generation_prompt(dataset_name: str, model_name: str, instance: dict[str, str], seed: int) -> list[dict]:
    """ Get the initial generation prompt (conversation with few-shot examples) for the given instance. """
    
    if dataset_name == "fldx2_symbol":
        return get_fld_initial_generation_prompt(model_name, instance, seed=seed)
    elif dataset_name == "metamathqa_isabelle":
        return get_isabelle_initial_generation_prompt(model_name, instance)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    # test case
    prompt_example = get_initial_generation_prompt(
        dataset_name="fldx2_symbol", model_name="meta-llama/Llama-3.1-8B-Instruct",
        instance={"hypothesis_formula": "({AB}{c} & ¬{B}{c})", "facts_formula": """fact1: (x): ¬{D}x -> ({A}x & {C}x)"""},
        seed=0,
    )
    for d in prompt_example:
        print(d)
