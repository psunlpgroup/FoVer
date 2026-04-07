import json
import random

from src.config import splits_list
from src.path import get_error_labels_path
from src.typing import ErrorLabelInstance
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    generate_statement_and_proof import IsabelleInformalToFormalTap, \
        get_generated_formal_proofs_thy_file_path, \
        get_generated_formal_proof_path
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    get_few_shot_prompt import get_formal_statement_and_proof_from_full_theorem
from src.dataset_creation.base_dataset_specific.isabelle.\
    error_detection.preprocessing import get_formal_theorems_for_error_detection_dir, \
        is_proof_valid_format
from src.dataset_creation.base_dataset_specific.isabelle.\
    error_detection.error_detection import is_isabelle_result_including_error
from src.llm.utils import save_md5_hash
from src.dataset_creation.utils import get_error_labels_stats


def get_steps_from_proof(proof: str, include_lemma: bool=False) -> list[str] | None:
    """ Get the steps from the proof. """
    
    # check if the proof is valid format
    if not is_proof_valid_format(proof):
        return None
    # now we know that the proof starts with "proof -" and ends with "qed"
    
    # derive expected number of steps based on the two-line-per-step structure
    raw_lines = proof.splitlines()
    number_of_steps = (len(raw_lines) - 2) // 2
    
    # drop "proof -" and "qed" lines, keep original indentation
    body_lines = raw_lines[1:-1]
    
    # each step occupies two lines (statement + justification)
    if len(body_lines) % 2 != 0:
        return None
    
    steps: list[str] = []
    for idx in range(0, len(body_lines), 2):
        statement_line = body_lines[idx]
        justification_line = body_lines[idx + 1]
        
        if "by" not in justification_line:
            return None
        
        if include_lemma:
            steps.append(" ".join([statement_line.strip(), justification_line.strip()]))
        else:
            steps.append(statement_line.strip())
    
    if len(steps) != number_of_steps:
        return None
    
    return steps


def get_isabelle_step_level_error_labels(
        dataset_name: str, initial_generation_model_name: str,
        conversion_model_name: str, formal_proof_model_name: str,
        split: str, data_id: str,
        proof_steps: list[str]
    ) -> list[bool] | None:
    """ Get the step-level error labels for the proof. """
    
    proof_dir = get_formal_theorems_for_error_detection_dir(
        dataset_name=dataset_name,
        initial_generation_model_name=initial_generation_model_name,
        conversion_model_name=conversion_model_name,
        formal_proof_model_name=formal_proof_model_name,
        split=split, data_id=data_id
    )
    
    # detect syntax error
    all_sorry_result_path = proof_dir / "all_sorry.result.json"
    if not all_sorry_result_path.exists():
        print(f"Result file not found at {all_sorry_result_path}.")
        return None
    
    try:
        with open(all_sorry_result_path, "r") as f:
            all_sorry_proof: dict = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {all_sorry_result_path}.")
        return None
    
    if not all_sorry_proof["success"]:
        # syntax error detected
        print(f"Syntax error detected for {data_id} at {all_sorry_result_path}.")
        return None
    
    # detect if all steps are correct
    all_sledgehammer_result_path = proof_dir / "all_sledgehammer.result.json"
    if all_sledgehammer_result_path.exists():
        all_sledgehammer_proof: dict | None = None
        try:
            with open(all_sledgehammer_result_path, "r") as f:
                all_sledgehammer_proof = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {all_sledgehammer_result_path}.")
        
        if all_sledgehammer_proof is not None:
            if all_sledgehammer_proof["success"]:
                # all steps are correct
                print(f"All steps are correct for {data_id} at {all_sledgehammer_result_path}.")
                return [True] * len(proof_steps)
    else:
        print(f"Result file not found at {all_sledgehammer_result_path}.")
    
    # get step level error labels
    error_labels: list[bool] = []
    for step_idx in range(len(proof_steps)):
        step_file_path = proof_dir / f"{step_idx:03}.result.json"
        if not step_file_path.exists():
            # error detection is incomplete
            print(f"Result file not found at {step_file_path}.")
            return None
        
        try:
            with open(step_file_path, "r") as f:
                step_proof: dict = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {step_file_path}.")
            return None
        
        # error
        if is_isabelle_result_including_error(step_proof):
            return None
        
        error_labels.append(step_proof["success"])
    
    return error_labels


def postprocess_last_step_of_isabelle_verification(
        proof_steps: list[str], proof_step_correctness: list[bool]
    ) -> tuple[list[str], list[bool]]:
    """ Postprocess the last step of Isabelle verification results. """
    
    is_all_process_correct = all(proof_step_correctness)
    
    # we remove the last step, which includes "thus ?thesis"
    if "thus ?thesis" not in proof_steps[-1]:
        raise ValueError(
            "The last step should include 'thus ?thesis' in Isabelle."
        )
    proof_steps = proof_steps[:-1]
    proof_step_correctness = proof_step_correctness[:-1]
    
    # the last step is updated to is_all_process_correct
    proof_step_correctness[-1] = is_all_process_correct
    
    # the last step is updated to include "The final answer is"
    if len(proof_steps) > 1 and "then have" not in proof_steps[-1]:
        raise ValueError(
            f"The last step should include 'then have' in Isabelle, if there are multiple steps. Found: {proof_steps[-1]}"
        )
    
    selected_message = random.Random(proof_steps[-1]).choice(
        ["The final answer is", "Therefore, the final answer is", "Thus, the final answer is",
         "The answer is", "Therefore, the answer is", "Thus, the answer is"]
    )
    proof_steps[-1] = proof_steps[-1].replace("then have", selected_message)
    
    # assertion
    assert len(proof_steps) == len(proof_step_correctness)
    
    return proof_steps, proof_step_correctness


def main():
    args = IsabelleInformalToFormalTap().parse_args()
    
    path_list_for_printing = []
    
    for split in splits_list:
        # get converted formal proofs path
        converted_formal_proofs_path = \
            get_generated_formal_proof_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                formal_proof_model_name=args.formal_proof_generation_model_name,
                split=split
            )
        
        with open(converted_formal_proofs_path, "r") as f:
            converted_formal_proofs = [json.loads(line) for line in f]
        
        # postprocessing
        error_labels_list: list[ErrorLabelInstance] = []
        for d in converted_formal_proofs:
            # load formal theorem
            thy_file_path = get_formal_theorems_for_error_detection_dir(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                formal_proof_model_name=args.formal_proof_generation_model_name,
                split=split, data_id=d["id"]
            ) / "all_lemmas.thy"
            
            if not thy_file_path.exists():
                print(f"Theory file does not exist: {thy_file_path}. Skipping...")
                continue
            
            with open(thy_file_path, "r") as f:
                formal_theorem = f.read()
            
            try:
                formal_statement, formal_proof = get_formal_statement_and_proof_from_full_theorem(
                    formal_theorem
                )
            except Exception as e:
                print(f"Error at get_formal_statement_and_proof_from_full_theorem: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            proof_steps = get_steps_from_proof(formal_proof, include_lemma=True)
            
            if proof_steps is None:
                print(f"Invalid proof format for {d['id']} (detected at get_steps_from_proof). Skipping...")
                continue
            
            # get step-level error label
            proof_step_correctness = get_isabelle_step_level_error_labels(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                formal_proof_model_name=args.formal_proof_generation_model_name,
                split=split, data_id=d["id"],
                proof_steps=proof_steps,
            )
            
            if proof_step_correctness is None:
                print(f"Verification result is not available for {d['id']}. Skipping...")
                continue
            
            proof_steps, proof_step_correctness = postprocess_last_step_of_isabelle_verification(
                proof_steps=proof_steps, proof_step_correctness=proof_step_correctness
            )
            
            model_short_name = args.base_model_name.split("/")[-1]
            error_label_d: ErrorLabelInstance = {
                "id": f"{d['id']}_{model_short_name}",
                "problem": formal_statement,
                "proof_steps": proof_steps,
                "all_process_correct": all(proof_step_correctness),
                "proof_step_correctness": proof_step_correctness
            }
            
            error_labels_list.append(error_label_d)
        
        # get error labels path
        error_labels_path = get_error_labels_path(
            dataset_name=f"isabelle_{args.dataset_name}",
            model_name=args.base_model_name,
            split=split, seed="selected"
        )
        error_labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_labels_path, "w") as f:
            for error_label in error_labels_list:
                f.write(json.dumps(error_label) + "\n")
        save_md5_hash(error_labels_path)
        
        # save stats
        stats = get_error_labels_stats(error_labels_list)
        stats_path = error_labels_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
    
        path_list_for_printing.append(str(error_labels_path))
    
    print("Error detection postprocessing completed.")
    for path_for_printing in path_list_for_printing:
        print(f"Saved error labels at: {path_for_printing}")


if __name__ == "__main__":
    main()
