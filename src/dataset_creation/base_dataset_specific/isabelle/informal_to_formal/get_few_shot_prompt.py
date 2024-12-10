from pathlib import Path
import json
import random

from src.dataset_creation.prompts import get_assistant_message, get_user_message
from src.downstream_evaluation.utils.postprocess \
    import get_solution_steps_from_response


isabelle_few_shot_examples_dir = Path(
    "src/dataset_creation/base_dataset_specific/" \
        "isabelle/conversion_few_shot_examples"
)


input_format_for_statement_conversion = """Your task is to convert the following informal statement into a formal statement in Isabelle 2022.
* Make your formal statement consistent with the provided Informal Statement.
* The final answer can be wrong, but your formal statement should be faithful to the informal statement and should not correct the mistakes in the informal statement.
* You should not use new variables in "shows".
* The "shows" part is expected to be formatted as "variable = number" (e.g., "x = 3").

** Informal Statement **
{informal_statement}"""


input_format_for_proof_conversion = """Your task is to convert the following informal proof into a formal proof in Isabelle 2022.
* The input informal proof can be wrong, but your formal proof should be faithful to the informal proof and should not correct the mistakes in the informal proof.
* In your formal proof, use variables defined in the provided Formal Statement.
* Use sledgehammer.
※ You should use defined variables whenever possible and should not write equations that only contain numbers.
* The last step is expected to be the same as the equation shown in the “shows” section of the Formal Statement.
* Include informal statements and proof as comments.

** Informal Statement **
{informal_statement}

** Informal Proof **
{informal_proof}

** Formal Statement **
{formal_statement}"""


def get_converted_informal_statement_and_proof(
        informal_statement: str, informal_proof: str
    ) -> tuple[str, str]:
    
    informal_steps_list = get_solution_steps_from_response(informal_proof)
    
    final_answer_statement = informal_steps_list[-1].replace("Therefore, the", "The"
        ).replace("(arabic numerals) ", "")
    processed_informal_statement = "\n".join(
        [informal_statement, "Final Answer -- " + final_answer_statement]
    )
    
    processed_informal_proof = "\n".join(informal_steps_list)
    
    return processed_informal_statement, processed_informal_proof


def get_input_format_for_statement_conversion(
        informal_statement: str, informal_proof: str) -> str:
    
    processed_informal_statement, _ = \
        get_converted_informal_statement_and_proof(
            informal_statement, informal_proof
        )
    
    return input_format_for_statement_conversion.format(
        informal_statement=processed_informal_statement
    )


def get_input_format_for_proof_conversion(
        informal_statement: str, informal_proof: str,
        formal_statement: str) -> str:

    processed_informal_statement, processed_nformal_proof = \
        get_converted_informal_statement_and_proof(
            informal_statement, informal_proof
        )
    
    return input_format_for_proof_conversion.format(
        informal_statement=processed_informal_statement,
        informal_proof=processed_nformal_proof,
        formal_statement=formal_statement
    )


def get_formal_statement_from_full_theorem(full_theorem: str) -> str:
    statement, _ = full_theorem.split("\nproof -", 1)
    return statement


def get_informal_statements_and_proof_dict(dataset_name: str) -> \
        dict[str, dict]:
    
    informal_statements_and_proofs_path = isabelle_few_shot_examples_dir / \
        "informal_statements_and_proofs.jsonl"
    with open(informal_statements_and_proofs_path, "r") as f:
        informal_statements_and_proofs = [json.loads(line) for line in f]
    
    informal_statements_and_proofs_dict = {}
    for example in informal_statements_and_proofs:
        if dataset_name in example["id"]:
            informal_statements_and_proofs_dict[example["id"]] = example

    return informal_statements_and_proofs_dict


def get_formal_proofs_for_few_shot_examples(dataset_name: str) \
        -> list[tuple[Path, str]]:
    # get file paths for formal proofs
    examples_files: list[Path] = []
    for examples_category in ["correct_proofs", "incorrect_proofs"]:
        examples_dir = isabelle_few_shot_examples_dir / examples_category
        
        # all files start with dataset_name
        examples_files.extend(examples_dir.glob(f"{dataset_name}*"))
    
    # shuffle examples_files
    examples_files = random.Random(68).sample(
        sorted(examples_files), len(examples_files)
    )
    
    formal_proofs = []
    for example_file in examples_files:
        # load formal proof
        with open(example_file, "r") as f:
            formal_proof = "".join(f.readlines())
        
        formal_proofs.append((example_file, formal_proof))
    
    return formal_proofs


def load_few_shot_examples_for_statement_conversion(
        dataset_name: str, model_name: str) -> list[dict]:

    few_shot_formal_proofs = get_formal_proofs_for_few_shot_examples(
        dataset_name
    )
    
    informal_statements_and_proofs_dict = \
        get_informal_statements_and_proof_dict(dataset_name)
    
    few_shot_examples: list[dict] = []
    for example_file, formal_proof in few_shot_formal_proofs:
        data_id = example_file.stem
        
        # make informal statement and proof
        informal_data = informal_statements_and_proofs_dict[data_id]
        informal_input = \
            input_format_for_statement_conversion.format(
                informal_statement=informal_data["question"]
            )
        
        # append to few_shot_examples
        few_shot_examples.append(
            get_user_message(informal_input)
        )
        
        # append to few_shot_examples
        formal_statement = get_formal_statement_from_full_theorem(formal_proof)
        few_shot_examples.append(
            get_assistant_message(formal_statement, model_name)
        )
    
    return few_shot_examples


def load_few_shot_examples_for_proof_conversion(
        dataset_name: str, model_name: str) -> list[dict]:
    
    few_shot_formal_proofs = get_formal_proofs_for_few_shot_examples(
        dataset_name
    )
    
    # make few-shot examples
    informal_statements_and_proofs_dict = \
        get_informal_statements_and_proof_dict(dataset_name)
    
    few_shot_examples: list[dict] = []
    for example_file, formal_proof in few_shot_formal_proofs:
        data_id = example_file.stem
        
        formal_statement = get_formal_statement_from_full_theorem(
            formal_proof
        )
        
        # make informal statement and proof
        informal_data = informal_statements_and_proofs_dict[data_id]
        informal_input = \
            get_input_format_for_proof_conversion(
                informal_data["question"], informal_data["response"],
                formal_statement=formal_statement
            )
        
        # append to few_shot_examples
        few_shot_examples.append(
            get_user_message(informal_input)
        )
        
        # append to few_shot_examples
        few_shot_examples.append(
            get_assistant_message(formal_proof, model_name)
        )
    
    return few_shot_examples


new_question_instruction = "Your response to the following question should follow the format (e.g., structure, style, line breaks) of responses in previous examples.\n\n"""


def get_few_shot_prompt_for_statement_conversion(
        dataset_name: str, model_name: str,
        informal_data: dict) -> list[dict]:
    
    if dataset_name == "metamathqa_gsm8k":
        dataset_name = "gsm8k"
    
    few_shot_examples = \
        load_few_shot_examples_for_statement_conversion(
            dataset_name, model_name
        )
    
    if len(few_shot_examples) == 0:
        raise ValueError(f"No few-shot examples found for {dataset_name}.")
    
    new_input = new_question_instruction + \
        get_input_format_for_statement_conversion(
            informal_statement=informal_data["question"],
            informal_proof=informal_data["response"]
        )
    
    conversation = few_shot_examples + [get_user_message(new_input)]
    
    return conversation


def get_few_shot_prompt_for_proof_conversion(
        dataset_name: str, model_name: str,
        informal_data: dict, formal_statement: str
    ) -> list[dict]:

    if dataset_name == "metamathqa_gsm8k":
        dataset_name = "gsm8k"
    
    few_shot_examples = \
        load_few_shot_examples_for_proof_conversion(
            dataset_name, model_name
        )
    
    if len(few_shot_examples) == 0:
        raise ValueError(f"No few-shot examples found for {dataset_name}.")
    
    new_input = new_question_instruction + \
        get_input_format_for_proof_conversion(
            informal_data["question"], informal_data["response"],
            formal_statement=formal_statement
        )
    
    conversation = few_shot_examples + [get_user_message(new_input)]
    
    return conversation


if __name__ == "__main__":
    few_shot_examples = \
        load_few_shot_examples_for_proof_conversion(
            "gsm8k", "meta-llama/Llama-3.1-8B-Instruct"
        )
    
    for example in few_shot_examples:
        print()
        print("=====")
        print(example)
        print(example["content"])
        print("=====")
