import json

from src.typing import PRM_PRED
from src.path import get_direct_evaluation_outputs_path, get_direct_evaluation_metrics_path
from src.direct_evaluation.run_direct_evaluation import DirectEvaluationTap
from src.utils.datasets import load_dataset
from src.utils.prm import postprocess_prm_output, postprocess_prm_output_from_vllm_reward_model
from src.utils.evaluation import get_binary_evaluation_metrics, get_float_evaluation_metrics, get_threshold_for_f1
from src.downstream_evaluation.sample_and_rank.postprocess_verification_outputs_sota_prms import get_verification_from_hidden_states_of_causal_model


def convert_y_to_binary(y_list: list[PRM_PRED]) -> list[int]:
    # False ("incorrect") is the positive class -> convert to 1
    convert_dict = {
        True: 0,
        False: 1,
        None: 0,  # treat None prediction as "correct" prediction
    }
    
    return [convert_dict[y] for y in y_list]


def get_metrics_dict(y_true: list[bool], y_pred: list, threshold: float | None = None) -> dict:
    if type(y_pred[0]) in [PRM_PRED, bool]:
        y_true_binary = convert_y_to_binary(y_true)
        y_pred_binary = convert_y_to_binary(y_pred)
        
        return get_binary_evaluation_metrics(y_true_binary, y_pred_binary)
    elif type(y_pred[0]) == float:
        return get_float_evaluation_metrics(y_true, y_pred, threshold=threshold)
    else:
        return {"error": f"Unknown type: {type(y_pred[0])}"}


def main():
    args = DirectEvaluationTap().parse_args()
    
    splits_list = ["test"]
    if "fover" in args.dataset_name:  # our dataset
        splits_list.append("train")
    
    for split in splits_list:
        # load dataset
        dataset = load_dataset(args.dataset_name, split=split)
        if args.max_num_evaluation_instances is not None:
            if len(dataset) > args.max_num_evaluation_instances:
                dataset = dataset.shuffle(seed=68)
                dataset = dataset.select(range(args.max_num_evaluation_instances))
        
        print(f"Loaded {len(dataset)} instances from {args.dataset_name} dataset for {split} split.")

        # load evaluation outputs
        evaluation_output_path = get_direct_evaluation_outputs_path(
            dataset_name=args.dataset_name, model_name=args.verification_model_name, split=split, prompt_type=args.verification_prompt_type)
        with open(evaluation_output_path, "r") as f:
            raw_evaluation_outputs = [json.loads(line) for line in f]
        
        assert len(dataset) == len(raw_evaluation_outputs), f"Length mismatch: {len(dataset)} != {len(raw_evaluation_outputs)}, {evaluation_output_path}"
        
        # load logprobs
        if args.verification_prompt_type == "zero-shot":
            logprobs_path = evaluation_output_path.with_suffix(".logprobs.jsonl")
            if logprobs_path.exists():
                with open(logprobs_path, "r") as f:
                    logprobs = [json.loads(line) for line in f]
            else:
                raise FileNotFoundError(f"Logprobs file not found: {logprobs_path}")

            evaluation_outputs = raw_evaluation_outputs
        elif args.verification_prompt_type == "multi-turn":
            logprobs = None

            if args.verification_model_name in [
                                    "Qwen/Qwen2.5-Math-7B-PRM800K",
                                    "Qwen/Qwen2.5-Math-PRM-7B",
                                    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
                                ]:
                evaluation_outputs = raw_evaluation_outputs
            else:
                evaluation_outputs = get_verification_from_hidden_states_of_causal_model(
                    verification_model_name=args.verification_model_name,
                    raw_verification_outputs=raw_evaluation_outputs,
                )
        else:
            raise ValueError(f"Unknown prompt type: {args.verification_prompt_type}")
        
        # postprocess evaluation outputs
        y_pred_step_level_all: list = []
        y_true_step_level_all: list = []
        y_pred_step_level_not_flattened_all: list = [] # for logging purpose, not used for calculating metrics
        y_true_step_level_not_flattened_all: list = [] # for logging purpose, not used for calculating metrics
        y_pred_instance_level_all: list = []
        y_true_instance_level_all: list = []
        
        for idx in range(len(dataset)):
            assert dataset[idx]["id"] == evaluation_outputs[idx]["id"], f"ID mismatch: {dataset[idx]['id']} != {evaluation_outputs[idx]['id']}"
            
            instance = dataset[idx]
            evaluation_output = evaluation_outputs[idx]
            
            # postprocess evaluation output
            if args.verification_prompt_type == "zero-shot":
                evaluation = postprocess_prm_output(
                    verification_score_type="logprob_min",
                    original_y_true=instance["error_labels"],
                    evaluation_output=evaluation_output,
                    base_model_name=args.base_model_name,
                    logprobs=logprobs[idx]["logprobs"],
                    # ProcessBench includes None prediction for the steps after
                    # the first error step. We skip these steps.
                    remove_step_if_y_true_is_none=True,
                )
            elif args.verification_prompt_type == "multi-turn":
                evaluation = postprocess_prm_output_from_vllm_reward_model(
                    verification_output=evaluation_output,
                    original_y_true=instance["error_labels"],
                    verification_model=args.verification_model_name,
                    verification_score_type="logprob_min",  # dummy value
                    # ProcessBench includes None prediction for the steps after
                    # the first error step. We skip these steps.
                    remove_step_if_y_true_is_none=True,
                )
            else:
                raise ValueError(f"Unknown prompt type: {args.verification_prompt_type}")
            
            y_pred_step_level_all.extend(evaluation["y_pred_step_level"])
            y_true_step_level_all.extend(evaluation["y_true_step_level"])

            y_pred_step_level_not_flattened_all.append(evaluation["y_pred_step_level"])
            y_true_step_level_not_flattened_all.append(evaluation["y_true_step_level"])

            y_pred_instance_level_all.append(evaluation["y_pred_instance_level"])
            y_true_instance_level_all.append(evaluation["y_true_instance_level"])
        
        # # this is for the old version of our process. the new version does not
        # # face this issue.
        # # ---
        # # filter out predictions
        # # in our framework, baseline models that are not fine-tuned on our
        # # dataset can make mistakes in output format.
        # if args.verification_model_name == args.base_model_name:
        #     # save y_pred
        #     baseline_y_pred_dict = {
        #         "y_pred_step_level": y_pred_step_level_all,
        #         "y_pred_instance_level": y_pred_instance_level_all,
        #     }
        # else:
        #     baseline_metrics_path = get_direct_evaluation_metrics_path(
        #         dataset_name=args.dataset_name,
        #         # base model for verification
        #         verification_model_name=args.base_model_name,
        #         split=split,
        #         verification_prompt_type=args.verification_prompt_type
        #     )

        #     # load baseline y_pred
        #     with open(baseline_metrics_path, "r") as f:
        #         baseline_metrics_path = json.load(f)
            
        #     baseline_y_pred_dict = {
        #         "y_pred_step_level": baseline_metrics_path[
        #             "y_pred_step_level"],
        #         "y_pred_instance_level": baseline_metrics_path[
        #             "y_pred_instance_level"],
        #     }
        
        # # filter out none predictions by baseline
        # y_pred_step_level_all = [
        #     y_pred
        #     for idx, y_pred in enumerate(y_pred_step_level_all)
        #     if baseline_y_pred_dict["y_pred_step_level"][idx] is not None
        # ]
        # y_true_step_level_all = [
        #     y_true
        #     for idx, y_true in enumerate(y_true_step_level_all)
        #     if baseline_y_pred_dict["y_pred_step_level"][idx] is not None
        # ]

        # y_pred_instance_level_all = [
        #     y_pred
        #     for idx, y_pred in enumerate(y_pred_instance_level_all)
        #     if baseline_y_pred_dict["y_pred_instance_level"][idx] is not None
        # ]
        # y_true_instance_level_all = [
        #     y_true
        #     for idx, y_true in enumerate(y_true_instance_level_all)
        #     if baseline_y_pred_dict["y_pred_instance_level"][idx] is not None
        # ]

        # y_pred_step_level_not_flattened_all = [
        #     y_pred
        #     for idx, y_pred in enumerate(y_pred_step_level_not_flattened_all)
        #     if baseline_y_pred_dict["y_pred_instance_level"][idx] is not None
        # ]
        # y_true_step_level_not_flattened_all = [
        #     y_true
        #     for idx, y_true in enumerate(y_true_step_level_not_flattened_all)
        #     if baseline_y_pred_dict["y_pred_instance_level"][idx] is not None
        # ]

        # get metrics
        num_all_steps = len(y_true_step_level_all)
        
        # get thresholds
        # we ues best F1 threshold for gsm8k
        # as in the ProcessBench paper https://arxiv.org/abs/2412.06559
        if "processbench" not in args.dataset_name:
            # we do not evaluate f1 for other datasets
            instance_threshold = None
            step_threshold = None
        elif args.dataset_name == "direct_evaluation_datasets/processbench_gsm8k":
            instance_threshold = get_threshold_for_f1(
                y_true=y_true_instance_level_all,
                y_pred=y_pred_instance_level_all
            )
            step_threshold = get_threshold_for_f1(
                y_true=y_true_step_level_all,
                y_pred=y_pred_step_level_all
            )
        else:
            # other processbench datasets
            evaluation_metrics_path = get_direct_evaluation_metrics_path(
                dataset_name="direct_evaluation_datasets/processbench_gsm8k",
                verification_model_name=args.verification_model_name,
                split=split, verification_prompt_type=args.verification_prompt_type
            )

            with open(evaluation_metrics_path, "r") as f:
                gsm8k_metrics = json.load(f)
            instance_threshold = gsm8k_metrics["performance"][
                "instance_level"]["threshold"]
            step_threshold = gsm8k_metrics["performance"][
                "step_level"]["threshold"]

        # get metrics
        metrics = {
            "performance": {
                "instance_level": get_metrics_dict(
                    y_true_instance_level_all, y_pred_instance_level_all,
                    threshold=instance_threshold
                ),
                "step_level": get_metrics_dict(
                    y_true_step_level_all, y_pred_step_level_all,
                    threshold=step_threshold
                ),
            },
            # "majority_label_baseline": {
            #     "instance_level": get_metrics_dict(y_true_instance_level_all, [True] * len(y_true_instance_level_all)),
            #     "step_level": get_metrics_dict(y_true_step_level_all, [True] * len(y_true_step_level_all)),
            # },
            "num_instances": len(dataset),
            "num_all_steps": num_all_steps,
            "y_pred_step_level": y_pred_step_level_all,
            "y_true_step_level": y_true_step_level_all,
            "y_pred_step_level_not_flattened": y_pred_step_level_not_flattened_all,
            "y_true_step_level_not_flattened": y_true_step_level_not_flattened_all,
            "y_pred_instance_level": y_pred_instance_level_all,
            "y_true_instance_level": y_true_instance_level_all,
        }
        
        # metrics
        evaluation_metrics_path = get_direct_evaluation_metrics_path(
            dataset_name=args.dataset_name,
            verification_model_name=args.verification_model_name,
            split=split, verification_prompt_type=args.verification_prompt_type
        )
        evaluation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(evaluation_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
