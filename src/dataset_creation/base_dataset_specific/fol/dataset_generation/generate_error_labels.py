""" This code automatically verifies the outputs of LLMs to annotate error labels. """

import json
import traceback
import signal

from tqdm import tqdm

from src.config import splits_list
from src.path import get_base_dataset_path, get_prompt_for_initial_generation_path, get_initial_answers_path, get_error_labels_path
from src.typing import ErrorLabelInstance
from src.llm.utils import save_md5_hash
from src.dataset_creation.utils import get_error_labels_stats
from src.dataset_creation.base_dataset_specific.fol.verifier import verify_fld_proof_steps
from src.dataset_creation.initial_answer_generation.generate_initial_answers import DatasetCreationTap


class GenerateVerificationDatasetTap(DatasetCreationTap):
    model_name: str = "ground_truth"
    """ BASE_MODEL | Literal["ground_truth"]; 
    The model name for which the verification dataset is generated.
    Default is "ground_truth", which use the ground truth proof steps. """


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("The code took too long to execute!")


if __name__ == "__main__":
    args = GenerateVerificationDatasetTap().parse_args()

    for split in splits_list[::-1]:
        # load the base dataset
        base_dataset_path = get_base_dataset_path(dataset_name=args.dataset_name, split=split)
        with open(base_dataset_path, "r") as f:
            base_dataset_list = [json.loads(line) for line in f]
        
        # we can use prompts for any models because we only use the prompt for the problem description in this code
        prompt_model_name = args.model_name if args.model_name != "ground_truth" else "meta-llama/Llama-3.1-8B-Instruct"
        
        if args.model_name == "ground_truth":
            num_samples = 1
        else:
            num_samples = args.num_samples
        
        for seed in range(1, num_samples + 1):
            # load the initial prompts
            initial_prompts_path = get_prompt_for_initial_generation_path(
                dataset_name=args.dataset_name, model_name=prompt_model_name,
                split=split, seed=seed
            )
            with open(initial_prompts_path, "r") as f:
                initial_prompts_list = [json.loads(line) for line in f]
            
            # load the initial answers
            initial_answers_path = get_initial_answers_path(
                dataset_name=args.dataset_name, model_name=args.model_name,
                split=split, seed=seed
            )
            if args.model_name == "ground_truth":
                initial_answers_list: list[dict] = []
                for base_data in base_dataset_list:
                    # preprocess the ground truth proof
                    proof = base_data["proof_formula"]
                    proof_label = base_data["proof_label"]
                    dummy_response = f"$proof$:\n{proof}\n\n$proof_label$: {proof_label}"
                    
                    initial_answers_list.append({"id": base_data["id"], "response": dummy_response})
            else:
                # load the initial answers
                with open(initial_answers_path, "r") as f:
                    initial_answers_list = [json.loads(line) for line in f]
            
            # verify the initial answers
            verification_results: list[ErrorLabelInstance] = []
            for idx in tqdm(range(len(initial_answers_list))):
                base_data = base_dataset_list[idx]
                prompt = initial_prompts_list[idx]
                initial_answer = initial_answers_list[idx]
                
                data_id = initial_answer["id"]
                assert data_id == base_data["id"], f"Data ID mismatch: {data_id} != {base_data['id']}"
                assert data_id == prompt["id"], f"Data ID mismatch: {data_id} != {prompt['id']}"
                
                # verify the proof steps
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(180)  # 180 seconds = 3 minutes
                try:
                    verification_result_dict = verify_fld_proof_steps(
                        model_output=initial_answer["response"],
                        facts_formula=base_data["facts_formula"], hypothesis_formula=base_data["hypothesis_formula"], y_true=base_data["proof_label"]
                    )
                    signal.alarm(0)
                except Exception as e:
                    print(f"Error for verifying output on {data_id} by {args.model_name}, skipping...")
                    full_traceback = traceback.format_exc()
                    if "Proof is empty" not in full_traceback and "Model output is too long" not in full_traceback:
                        # unexpected error
                        print(full_traceback)
                    continue
                
                # assertion for the ground truth input
                if args.model_name == "ground_truth":
                    if not all(verification_result_dict["proof_step_correctness"]):
                        print("Proof step correctness is not all True for the ground truth input.\n" + \
                            f"data_id: {data_id}\n" + \
                            f"proof_step_correctness: {verification_result_dict['proof_step_correctness']}"
                        )
                        continue

                    if not verification_result_dict["is_proof_consistent_to_y_pred"]:
                        print("Proof is not consistent with the predicted proof label for the ground truth input.\n" + \
                            f"data_id: {data_id}\n" + \
                            f"is_proof_consistent_to_y_pred: {verification_result_dict['is_proof_consistent_to_y_pred']}"""
                        )
                        continue
                
                # add final conclusion to the proof steps
                verification_result_dict["proof_steps"].append(
                    f"The final answer is {verification_result_dict['y_pred']}"
                )
                verification_result_dict["proof_step_correctness"].append(
                    verification_result_dict["y_correct"]
                )
                
                # save the verification results
                model_short_name = args.model_name.split("/")[-1]
                verification_results.append(
                    {
                        "id": f"{data_id}_{model_short_name}",
                        "base_data_id": data_id,
                        "y_true": base_data["proof_label"],
                        "y_pred": verification_result_dict["y_pred"],
                        "y_correct": verification_result_dict["y_correct"],
                        "all_process_correct": all(verification_result_dict["proof_step_correctness"]),
                        "proof_step_correctness": verification_result_dict["proof_step_correctness"],
                        "is_proof_consistent_to_y_pred": verification_result_dict["is_proof_consistent_to_y_pred"],
                        "problem": prompt["prompt"][-1]["content"],
                        "proof_steps": verification_result_dict["proof_steps"],
                        "processed_proof_steps": verification_result_dict["processed_proof_steps"],
                        "hypothesis_formula": base_data["hypothesis_formula"],
                        "facts_formula": base_data["facts_formula"],
                    }
                )
            
            # save the verification results
            output_path = get_error_labels_path(
                dataset_name=args.dataset_name, model_name=args.model_name,
                split=split, seed=seed
            ).with_suffix(".full.jsonl")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                for d in verification_results:
                    f.write(json.dumps(d) + "\n")
            save_md5_hash(output_path)
            
            # get statistics
            stats = get_error_labels_stats(verification_results)
            
            stats_path = output_path.with_suffix(".stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
