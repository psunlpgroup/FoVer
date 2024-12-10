import json
from pathlib import Path
import shutil
import random

from src.path import intermediate_dir
from src.config import splits_list
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    convert_to_formal import IsabelleInformalToFormalTap, \
        get_converted_formal_statement_or_proofs_path


class IsabelleErrorDetectionPreprocessingTap(IsabelleInformalToFormalTap):
    num_parallel_processes: int = 64


isabelle_generated_thy_files_dir = intermediate_dir / "isabelle" \
    / "formal_proofs"

def get_formal_proofs_for_error_detection_dir(
        dataset_name: str, initial_generation_model_name: str,
        conversion_model_name: str, split: str, data_id: str) -> Path:
    """ Get the path to the thy file of the converted formal proofs for the
    given dataset, model, and split. """
    
    initial_model_short_name = initial_generation_model_name.split("/")[-1]
    conversion_model_short_name = conversion_model_name.split("/")[-1]
    return isabelle_generated_thy_files_dir / \
        dataset_name / f"initial_generation={initial_model_short_name}" / \
        f"conversion={conversion_model_short_name}" / split / data_id

def clean_up_isabelle_statement_and_proof(statement_or_proof: str) -> str:
    """ Remove comments from the proof. """
    lines = statement_or_proof.split("\n")
    cleaned_lines = []
    for line in lines:
        if "(*" in line and "*)" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def generate_proofs_for_each_step(proof: str) -> list[str]:
    """ Generate proofs for each step in the proof. """
    
    output_list = []
    num_steps = proof.count("sledgehammer")
    for step in range(num_steps):
        # keep step-th sledgehammer
        edited_proof = proof.replace("sledgehammer", "sorry", step)
        edited_proof = edited_proof.replace("sledgehammer", "keepthisline", 1)
        edited_proof = edited_proof.replace("sledgehammer", "sorry")
        edited_proof = edited_proof.replace("keepthisline", "sledgehammer")
        
        output_list.append(edited_proof)
    
    return output_list


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
        conversion_model_name: str, split: str, batch_idx: int
    ) -> Path:
    
    initial_short_name = initial_generation_model_name.split("/")[-1]
    conversion_short_name = conversion_model_name.split("/")[-1]
    
    return intermediate_dir / "isabelle" / "formal_proofs_batch_file_names" / \
        dataset_name / f"initial_generation={initial_short_name}" / \
        f"conversion={conversion_short_name}"/ \
        split / f"batch_{batch_idx:03}.json"


def main():
    args = IsabelleErrorDetectionPreprocessingTap().parse_args()
    
    for split in splits_list:
        ###
        # preprocess and save proofs for step-level error detection
        
        # load model generated proofs
        proof_path = get_converted_formal_statement_or_proofs_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
            statement_or_proof="proof", split=split
        )
        with open(proof_path, "r") as f:
            proofs = [json.loads(line) for line in f]
        
        # save proofs for each step
        for proof in proofs:
            # removce comments
            proof_dir = get_formal_proofs_for_error_detection_dir(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                split=split, data_id=proof["id"]
            )
            
            if proof_dir.exists():
                # remove all .thy files
                for file in proof_dir.glob("*.thy"):
                    file.unlink()
            else:
                proof_dir.mkdir(parents=True, exist_ok=True)
            
            proof_without_comments = clean_up_isabelle_statement_and_proof(
                proof["response"])

            proof_for_each_step = generate_proofs_for_each_step(
                proof_without_comments
            )
            
            for step, proof_for_step in enumerate(proof_for_each_step):
                proof_file_path = proof_dir / f"{step:03d}.thy"
                with open(proof_file_path, "w") as f:
                    f.write(proof_for_step)
            
            # to check if the proof is correct
            proof_file_path = proof_dir / "all_sledgehammer.thy"
            with open(proof_file_path, "w") as f:
                f.write(proof_without_comments)
            
            # to check syntax error, replace all sledgehammer with sorry
            proof_file_path = proof_dir / "all_sorry.thy"
            with open(proof_file_path, "w") as f:
                f.write(proof_without_comments.replace("sledgehammer", "sorry"))
        
        ###
        # make batch for parallel isabelle execution
        
        # load model generated proofs
        proof_path = get_converted_formal_statement_or_proofs_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
            statement_or_proof="proof", split=split
        )
        with open(proof_path, "r") as f:
            proofs = [json.loads(line) for line in f]
        
        # load all .thy files
        all_thy_files = []
        for proof in proofs:
            proof_dir = get_formal_proofs_for_error_detection_dir(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                split=split, data_id=proof["id"]
            )
            
            # get all .thy files
            thy_files_list = proof_dir.glob("*.thy")
            all_thy_files.extend(thy_files_list)
        
        # clean up batch directory if exists
        batch_file_names_dir = get_batch_file_names_path(
            dataset_name=args.dataset_name,
            initial_generation_model_name=args.base_model_name,
            conversion_model_name=args.conversion_model_name,
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
                split=split, batch_idx=batch_idx
            )
            batch_file_names_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(batch_file_names_path, "w") as f:
                json.dump(batches[batch_idx], f)
            
            batch_files_list.append(batch_file_names_path)


if __name__ == "__main__":
    main()
