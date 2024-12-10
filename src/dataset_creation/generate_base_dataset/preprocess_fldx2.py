""" This code will generate a base dataset, which remove data we don't use from the original dataset. """

import json

from tqdm import tqdm
import datasets

from src.config import splits_list, full_names_dict
from src.path import get_base_dataset_path
from src.llm.utils import save_md5_hash


if __name__ == "__main__":
    dataset_name = "fldx2_symbol"
    
    # process each split
    for split in splits_list:
        print(f"Processing {dataset_name} {split}...")
        
        ###
        # fldx2_symbol
        processed_dataset: list[dict] = []

        dataset_full_name = full_names_dict[dataset_name]
        dataset = datasets.load_dataset(dataset_full_name, split=split)

        # from src.dataset_creation.base_dataset_specific.fol.utils \
        #     import is_proof_label_correct
        from src.dataset_creation.base_dataset_specific.fol.utils \
            import preprocess_fld_ground_truth_proof
        
        # filter out data that is difficult to verify by verifiers
        for data_idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            d: dict[str, str] = d  # this line is just for type hinting
            
            # remove data with "fake_formula"
            if "fake_formula" in d["facts_formula"]:
                continue
            
            # check if the proof is empty
            # if the proof is empty, the proof label should be "UNKNOWN"
            if len(d["proofs_formula"]) > 0:
                proofs_formula = d["proofs_formula"][0]
            else:
                assert d["proof_label"] == "UNKNOWN", \
                    f"proof_label is {d['proof_label']}, " \
                        "but proofs_formula is empty."
                proofs_formula = ""
            
            # solution with assumptions (e.g., proof by contradiction) is
            # difficult to verify by verifiers
            if "assump1" in proofs_formula:
                continue

            row = {
                "facts_formula": d["facts_formula"].replace(
                    " fact", "\nfact"
                ),
                "hypothesis_formula": d["hypothesis_formula"],
                "proof_formula": preprocess_fld_ground_truth_proof(
                    proof_formula=proofs_formula,
                    proof_label=d["proof_label"]
                ),
                "proof_label": d["proof_label"],
                "id": f"fldx2-{split}-{data_idx:06d}",
            }
            
            processed_dataset.append(row)
        
        # save dataset
        output_path = get_base_dataset_path(
            dataset_name=dataset_name, split=split
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to {output_path}...")
        with open(output_path, "w") as f:
            for d in processed_dataset:
                f.write(json.dumps(d) + "\n")
        save_md5_hash(output_path)
