import json

from src.config import splits_list
from src.path import get_error_labels_path
from src.typing import ErrorLabelInstance
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    convert_to_formal import IsabelleInformalToFormalTap, \
        get_converted_formal_proofs_thy_file_path, \
        get_converted_formal_statement_or_proofs_path
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    get_few_shot_prompt import get_formal_statement_from_full_theorem
from src.dataset_creation.base_dataset_specific.isabelle.\
    error_detection.preprocessing import clean_up_isabelle_statement_and_proof, \
        get_formal_proofs_for_error_detection_dir
from src.dataset_creation.base_dataset_specific.isabelle.\
    error_detection.error_detection import is_isabelle_result_including_error
from src.llm.utils import save_md5_hash
from src.dataset_creation.utils import get_error_labels_stats


def get_steps_from_proof(proof: str) -> list[str] | None:
    """ Get the steps from the proof. """
    
    # use this for checking the number of steps
    number_of_steps = proof.count("sledgehammer")
    
    # clean up the proof (remove comments, etc.)
    proof = clean_up_isabelle_statement_and_proof(proof)
    
    # split into statement and proof
    if "proof -" not in proof:
        return None  # invalid format
    
    _, steps_str = proof.split("proof -", 1)
    
    # detect the last "qed" and remove the rest
    qed_idx = steps_str.rfind("qed")
    if qed_idx != -1:
        steps_str = steps_str[:qed_idx]
    
    sentences_list = steps_str.split("\n")
    sentences_list = [
        s for s in sentences_list
        if (len(s) > 0 and "sledgehammer" not in s)
    ]
    
    # clean up the steps
    steps = []
    for line in sentences_list:
        # remove the leading space
        while line[0] == " ":
            line = line[1:]
            if len(line) == 0:
                break
        
        # blank line
        if len(line) == 0:
            continue
        
        steps.append(line)
    
    # invalid if the number of lines are not equal to the number of steps
    if len(sentences_list) != number_of_steps:
        return None
    
    return steps


def get_isabelle_step_level_error_labels(
        dataset_name: str, initial_generation_model_name: str,
        conversion_model_name: str, split: str, data_id: str,
        proof_steps: list[str]
    ) -> list[bool] | None:
    """ Get the step-level error labels for the proof. """
    
    proof_dir = get_formal_proofs_for_error_detection_dir(
        dataset_name=dataset_name,
        initial_generation_model_name=initial_generation_model_name,
        conversion_model_name=conversion_model_name,
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


def main():
    args = IsabelleInformalToFormalTap().parse_args()
    
    for split in splits_list:
        # get converted formal proofs path
        converted_formal_proofs_path = \
            get_converted_formal_statement_or_proofs_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                statement_or_proof="proof", split=split
            )
        
        with open(converted_formal_proofs_path, "r") as f:
            converted_formal_proofs = [json.loads(line) for line in f]
        
        # postprocessing
        error_labels_list: list[ErrorLabelInstance] = []
        for d in converted_formal_proofs:
            # load formal theorem
            thy_file_path = get_formal_proofs_for_error_detection_dir(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                split=split, data_id=d["id"]
            ) / "all_sledgehammer.thy"
            
            with open(thy_file_path, "r") as f:
                formal_theorem = f.read()
            
            try:
                formal_statement = get_formal_statement_from_full_theorem(
                    formal_theorem
                )
            except Exception as e:
                print(f"Error at get_formal_statement_from_full_theorem: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            proof_steps = get_steps_from_proof(formal_theorem)
            
            if proof_steps is None:
                print(f"Invalid proof format for {d['id']} (detected at get_steps_from_proof). Skipping...")
                continue
            
            # get step-level error label
            proof_step_correctness = get_isabelle_step_level_error_labels(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                split=split, data_id=d["id"],
                proof_steps=proof_steps,
            )
            
            if proof_step_correctness is None:
                print(f"Verification result is not available for {d['id']}. Skipping...")
                continue
            
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


if __name__ == "__main__":
    main()
