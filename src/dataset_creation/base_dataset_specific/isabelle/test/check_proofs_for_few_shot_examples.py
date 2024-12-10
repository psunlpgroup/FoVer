# This code is for debugging purpose.
# This file is not used in the dataset creation pipeline.


from pathlib import Path

from src.dataset_creation.base_dataset_specific.isabelle.proof_checker.proof_checker import get_proof_checker


# few_shot_examples_dir = Path("src/dataset_creation/base_dataset_specific/isabelle/conversion_few_shot_examples/correct_proofs")
# few_shot_examples_dir = Path("src/dataset_creation/base_dataset_specific/isabelle/conversion_few_shot_examples/incorrect_proofs")
# few_shot_examples_dir = Path("model_responses/isabelle/formal_proofs/gsm8k/Llama-3.3-70B-Instruct/validation")
few_shot_examples_dir = Path("intermediate/isabelle/formal_proofs/gsm8k/Llama-3.3-70B-Instruct/train/gsm8k_train_00011")

def main():
    results_list: list[tuple[str, dict]] = []
    
    checker = get_proof_checker(port=9000)
    
    all_proof_files = few_shot_examples_dir.glob("*.thy")
    for proof_file in all_proof_files:
        if "005" not in proof_file.name:
            continue
        
        with open(proof_file, "r") as f:
            proof = f.read()
        result = checker.check(proof)
        print(result)
        # print("\n==== Success: %s" % result['success'])
        # print("--- Complete proof:\n%s" % result['theorem_and_proof'])
        
        results_list.append((proof_file, result))

    for file_name, result in results_list:
        if not result["success"]:
            print(f"\n=== Proof in {file_name} failed ===")
            print(result)
    
    if all(result["success"] for _, result in results_list):
        print("\nAll proofs passed!")


if __name__ == "__main__":
    main()
