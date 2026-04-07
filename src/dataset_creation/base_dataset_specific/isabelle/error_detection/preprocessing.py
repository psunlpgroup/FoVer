import json
from pathlib import Path
import shutil
import random
import copy

from src.path import intermediate_dir
from src.config import splits_list
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    generate_statement_and_proof import IsabelleInformalToFormalTap, \
        get_converted_formal_statement_or_proofs_path, \
        get_generated_formal_proof_path
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    utils import clean_up_isabelle_statement_and_proof
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    get_few_shot_prompt import get_formal_statement_and_proof_from_full_theorem


class IsabelleErrorDetectionPreprocessingTap(IsabelleInformalToFormalTap):
    num_parallel_processes: int = 64


isabelle_generated_thy_files_dir = intermediate_dir / "isabelle" \
    / "formal_proofs"

def get_formal_theorems_for_error_detection_dir(
        dataset_name: str, initial_generation_model_name: str,
        conversion_model_name: str, formal_proof_model_name: str,
        split: str, data_id: str) -> Path:
    """ Get the path to the thy file of the converted formal proofs for the
    given dataset, model, and split. """
    
    initial_model_short_name = initial_generation_model_name.split("/")[-1]
    conversion_model_short_name = conversion_model_name.split("/")[-1]
    formal_proof_model_short_name = formal_proof_model_name.split("/")[-1]
    
    return isabelle_generated_thy_files_dir / \
        dataset_name / f"initial_generation={initial_model_short_name}" / \
        f"conversion={conversion_model_short_name}" / \
        f"formal_proof_generation={formal_proof_model_short_name}" / \
            split / data_id


def is_proof_valid_format(raw_proof: str) -> bool:
    lines_list = raw_proof.split("\n")
    
    # the first line is "proof -"
    # the last line is "qed"
    # each step has 2 lines and lemma line is the second line in each step
    
    if "proof -" not in lines_list[0]:
        print(f"Expected 'proof -' in the first line but got: {lines_list[0]}")
        return False
    
    if "qed" not in lines_list[-1]:
        print(f"Expected 'qed' in the last line but got: {lines_list[-1]}")
        return False
    
    number_of_steps = (len(lines_list) - 2) // 2
    for step in range(number_of_steps):
        step_line_num = 1 + step * 2
        if step >= 1 and step != number_of_steps - 1:
            # after the second step and before the last step should be "then have"
            if "then have" not in lines_list[step_line_num]:
                print(
                    f"Expected 'then have' in line {step_line_num} but got: "
                    f"{lines_list[step_line_num]}"
                )
                return False

            # intermediate steps should not be "thus ?thesis"
            if "thus ?thesis" in lines_list[step_line_num]:
                print(
                    f"Did not expect 'thus ?thesis' in line {step_line_num} but got: "
                    f"{lines_list[step_line_num]}"
                )
                return False
        
        if step == number_of_steps - 1:
            # last step should be "thus ?thesis"
            if "thus ?thesis" not in lines_list[step_line_num]:
                print(
                    f"Expected 'thus ?thesis' in line {step_line_num} but got: "
                    f"{lines_list[step_line_num]}"
                )
                return False
        
        # lemma should include "by"
        lemma_line_num = step_line_num + 1
        if "by" not in lines_list[lemma_line_num]:
            print(f"Expected 'by' in line {lemma_line_num} but got: {lines_list[lemma_line_num]}")
            return False
    
    return True


def replace_step_lemma_with_sorry(raw_proof: str, step: int) -> str:
    """ Replace the step-th sledgehammer with sorry in the theorem. """
    
    lines_list = raw_proof.split("\n")
    
    # the first line is "proof -"
    # the last line is "qed"
    # each step has 2 lines and lemma line is the second line in each step
    step_line_num = 1 + step * 2 + 1
    
    # check
    if len(lines_list) <= step_line_num:
        raise ValueError(
            f"Step line number {step_line_num} exceeds the number of lines "
            f"in the proof: {len(lines_list)}"
        )
    
    if "by" not in lines_list[step_line_num]:
        raise ValueError(
            f"Expected 'by' in line {step_line_num} but got: "
            f"{lines_list[step_line_num]}"
        )
    
    # replace
    lines_list[step_line_num] = "        sorry"
    
    return "\n".join(lines_list)


def generate_theorems_for_each_step(raw_theorem: str) -> list[str]:
    """ Generate theorems for each step in the theorems. """
    
    statement, raw_proof = get_formal_statement_and_proof_from_full_theorem(
        raw_theorem
    )
    
    output_list = []
    num_steps = (len(raw_proof.split("\n")) - 2) // 2
    for keep_step in range(num_steps):
        # keep keep_step-th lemma
        edited_proof = copy.deepcopy(raw_proof)
        for step in range(num_steps):
            if step == keep_step:
                continue
            edited_proof = replace_step_lemma_with_sorry(edited_proof, step)
        
        # combine statement and proof
        edited_theorem = "\n".join([statement, edited_proof])
        output_list.append(edited_theorem)
    
    return output_list


def generate_all_sorry_theorem(raw_theorem: str) -> str:
    """ Generate the theorem with all sledgehammer replaced with sorry. """
    
    statement, raw_proof = get_formal_statement_and_proof_from_full_theorem(
        raw_theorem
    )
    
    # the first line is "proof -"
    # the last line is "qed"
    # each step has 2 lines and lemma line is the second line in each step
    num_steps = (len(raw_proof.split("\n")) - 2) // 2
    edited_proof = copy.deepcopy(raw_proof)
    for step in range(num_steps):
        edited_proof = replace_step_lemma_with_sorry(edited_proof, step)
    
    # combine statement and proof
    edited_theorem = "\n".join([statement, edited_proof])
    
    return edited_theorem


def make_batch_of_thy_files(
        thy_files: list[Path], batch_num: int) \
            -> dict[int, list[str]]:
    """ Make a batch of .thy files for parallel Isabelle execution """
    
    # make sure that all files for a signel theorem are in the same batch
    theorem_name_to_files: dict[str, list[Path]] = {}
    for file in thy_files:
        # parent directory is the theorem name
        theorem_name = str(file.parent)
        theorem_name_to_files.setdefault(theorem_name, []).append(file)
    
    # split theorem into batches
    theorem_names_list = sorted(list(theorem_name_to_files.keys()))
    theorem_names_list = random.Random(68).sample(
        theorem_names_list, len(theorem_names_list)
    )
    theorem_name_to_batch_dict: dict[str, int] = {}
    for theorem_idx, theorem_name in enumerate(theorem_names_list):
        batch_idx = theorem_idx % batch_num
        theorem_name_to_batch_dict[theorem_name] = batch_idx
    
    # split files into batches
    batches = {batch_idx: [] for batch_idx in range(batch_num)}
    for theorem_name in theorem_names_list:
        files_ = theorem_name_to_files[theorem_name]
        files_ = sorted(files_)
        
        files = []
        for file_ in files_:
            # put all_sorry in the front
            if "all_sorry" in str(file_):
                files = [file_] + files
            elif "all_sledgehammer" in str(file_):
                files = [file_] + files
            else:
                files.append(file_)
        
        batch_idx = theorem_name_to_batch_dict[theorem_name]
        batches[batch_idx].extend([str(file) for file in files])
    
    return batches


def get_batch_file_names_path(
        dataset_name: str, initial_generation_model_name: str,
        conversion_model_name: str, formal_proof_model_name: str,
        split: str, batch_idx: int
    ) -> Path:
    
    initial_short_name = initial_generation_model_name.split("/")[-1]
    conversion_short_name = conversion_model_name.split("/")[-1]
    formal_proof_short_name = formal_proof_model_name.split("/")[-1]
    
    return intermediate_dir / "isabelle" / "formal_proofs_batch_file_names" / \
        dataset_name / f"initial_generation={initial_short_name}" / \
        f"conversion={conversion_short_name}"/ \
        f"formal_proof_generation={formal_proof_short_name}" / \
        split / f"batch_{batch_idx:03}.json"


def main():
    args = IsabelleErrorDetectionPreprocessingTap().parse_args()
    
    for split in splits_list:
        ###
        # preprocess and save proofs for step-level error detection
        
        # load model generated statements and proofs
        statements_path = get_converted_formal_statement_or_proofs_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
            statement_or_proof="statement", split=split
        )
        with open(statements_path, "r") as f:
            statements = [json.loads(line) for line in f]
        
        proofs_path = get_generated_formal_proof_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
            formal_proof_model_name=args.formal_proof_generation_model_name,
            split=split
        )
        with open(proofs_path, "r") as f:
            proofs = [json.loads(line) for line in f]
        
        # check consistency
        if len(statements) != len(proofs):
            raise ValueError(
                "Number of statements and proofs do not match: "
                f"{len(statements)} statements vs {len(proofs)} proofs."
            )
        if not all(s["id"] == p["id"] for s, p in zip(statements, proofs)):
            raise ValueError(
                "IDs of statements and proofs do not match."
            )
        
        data_ids = [s["id"] for s in statements]
        
        # combine statement and proof
        theorems = [
            {
                "id": s["id"],
                "response": "\n".join([s["response"], p["response"]])
            }
            for s, p in zip(statements, proofs)
        ]
        
        # save proofs for each step
        for theorem in theorems:
            # removce comments
            theorem_dir = get_formal_theorems_for_error_detection_dir(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                formal_proof_model_name=args.formal_proof_generation_model_name,
                split=split, data_id=theorem["id"]
            )
            
            theorems_without_comments = clean_up_isabelle_statement_and_proof(
                theorem["response"]
            )
            
            # check format
            invalid_format = False
            try:
                _, raw_proof = get_formal_statement_and_proof_from_full_theorem(
                    theorems_without_comments
                )
            except Exception as e:
                print(f"Error in getting formal statement and proof: {e}")
                raw_proof = ""
                invalid_format = True
                
            if invalid_format or not is_proof_valid_format(raw_proof):
                print(f"Invalid proof format for theorem ID: {theorem['id']}")
                
                if theorem_dir.exists():
                    shutil.rmtree(theorem_dir)
                continue

            if theorem_dir.exists():
                # remove all .thy files
                for file in theorem_dir.glob("*.thy"):
                    file.unlink()
            else:
                theorem_dir.mkdir(parents=True, exist_ok=True)
            
            theorems_for_each_step = generate_theorems_for_each_step(
                theorems_without_comments
            )
            
            for step, theorem_for_step in enumerate(theorems_for_each_step):
                theorem_file_path = theorem_dir / f"{step:03d}.thy"
                with open(theorem_file_path, "w") as f:
                    f.write(theorem_for_step)
            
            # to check if the proof is correct
            theorem_file_path = theorem_dir / "all_lemmas.thy"
            with open(theorem_file_path, "w") as f:
                f.write(theorems_without_comments)
            
            # to check syntax error, replace all sledgehammer with sorry
            theorem_file_path = theorem_dir / "all_sorry.thy"
            all_sorry_theorem = generate_all_sorry_theorem(
                theorems_without_comments
            )
            with open(theorem_file_path, "w") as f:
                f.write(all_sorry_theorem)
        
        ###
        # make batch for parallel isabelle execution
        
        # load all .thy files
        all_thy_files = []
        for data_id in data_ids:
            theorem_dir = get_formal_theorems_for_error_detection_dir(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                formal_proof_model_name=args.formal_proof_generation_model_name,
                split=split, data_id=data_id
            )
            
            # get all .thy files
            thy_files_list = theorem_dir.glob("*.thy")
            all_thy_files.extend(thy_files_list)
        
        # clean up batch directory if exists
        batch_file_names_dir = get_batch_file_names_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
            formal_proof_model_name=args.formal_proof_generation_model_name,
            split=split, batch_idx=-1
        ).parent
        if batch_file_names_dir.exists():
            shutil.rmtree(batch_file_names_dir)
        
        # save batches
        batch_num = args.num_parallel_processes if split == "train" \
            else args.num_parallel_processes // 4
        batches = make_batch_of_thy_files(
            thy_files=all_thy_files, batch_num=batch_num
        )
        
        batch_files_list = []
        for batch_idx in range(batch_num):
            batch_file_names_path = get_batch_file_names_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                formal_proof_model_name=args.formal_proof_generation_model_name,
                split=split, batch_idx=batch_idx
            )
            batch_file_names_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(batch_file_names_path, "w") as f:
                json.dump(batches[batch_idx], f)
            
            batch_files_list.append(batch_file_names_path)


if __name__ == "__main__":
    main()
