import numpy as np
import torch

from src.typing import PRM_PRED


def get_verification_score_from_binary_prediction(
        y_pred_step_level: list[bool]) -> float:
    """ Get the verification score from the postprocessed output of the PRM. """
    # proportion of correct steps
    num_correct_steps = sum(
        [1 for y_pred in y_pred_step_level if y_pred is True]
    )
    num_steps = len(y_pred_step_level)
    return num_correct_steps / num_steps


def get_logprob_for_target_token(
        token_logprob_list: list[tuple[str, float]], token: str,
        return_pseudo_logprob_if_not_found=True) -> float | None:
    """ Get the log probability for a specific token. """
    
    for candidate_token, candidate_logprob in token_logprob_list:
        if candidate_token == token:
            # found!
            return candidate_logprob
    
    if return_pseudo_logprob_if_not_found:
        # if not found, return the log probability of the last token
        return token_logprob_list[-1][1]
    else:
        return None


def log_softmax(logprobs: list[float | None]) -> list[float]:
    """ Log softmax function. """
    
    logprobs_tensor = torch.tensor(logprobs)
    logprobs_tensor = torch.nn.functional.log_softmax(logprobs_tensor, dim=0)
    
    return logprobs_tensor.tolist()


def get_logprob_for_correct(
        logprobs: list[list[tuple[str, float]]],
        step_idx: int, base_model_name: str) -> float | None:
    """ Get the log probability for the correct prediction for a specific step. """
    
    if "Qwen" in base_model_name or "gemma" in base_model_name:
        # in qwen and gemma, each digit is separately tokenized
        step_tag_tokens = ["step", "_"] + list(str(step_idx))
        correct_token = "correct"
        incorrect_token = "incorrect"
    elif "Llama" in base_model_name:
        # in llama, number is represented in a single token
        step_tag_tokens = ["step", "_"] + [str(step_idx)]
        correct_token = "correct"
        incorrect_token = "incorrect"
    else:
        raise ValueError(f"Unknown base model name: {base_model_name}")
    
    # find the log probability for the correct prediction
    top_tokens = [pred[0][0] for pred in logprobs]
    
    # find the index of step_tag_tokens
    correct_log_prob = None
    incorrect_log_prob = None
    for i in range(len(top_tokens) - len(step_tag_tokens)):
        match = True
        for j in range(len(step_tag_tokens)):
            if top_tokens[i+j] != step_tag_tokens[j]:
                # no match
                match = False
                break
        
        if match:
            pred_idx = i + len(step_tag_tokens) + 1
            if len(top_tokens) <= pred_idx:
                # out of range
                break
            
            pred_label = top_tokens[pred_idx]
            if pred_label == correct_token or pred_label == incorrect_token:
                # found a match
                correct_log_prob = get_logprob_for_target_token(
                    logprobs[pred_idx], correct_token
                )
                incorrect_log_prob = get_logprob_for_target_token(
                    logprobs[pred_idx], incorrect_token
                )
                break
    
    if correct_log_prob is None:
        return None
    else:
        normalized_logprobs = log_softmax(
            [correct_log_prob, incorrect_log_prob]
        )
        
        return normalized_logprobs[0]


def get_postprocessed_prm_output_format(
        y_pred_step_level: list, y_true_step_level: list,
        verification_score_type: str) -> dict:
    """ Get the postprocessed output format for the PRM. """
    
    # check if all step-level predictions are valid
    is_y_pred_step_level_valid = [
        y_pred is not None for y_pred in y_pred_step_level
    ]
    is_all_step_scores_valid = all(is_y_pred_step_level_valid)
    is_all_step_scores_invalid = all(
        y_pred is None for y_pred in y_pred_step_level
    )
    invalid_steps_ratio = sum(
        y_pred is None for y_pred in y_pred_step_level
    ) / len(y_pred_step_level)
    
    ###
    # instance-level prediction
    if all([y_true is None for y_true in y_true_step_level]):
        # for downstream evaluation, we don't have the ground truth
        y_true_instance_level = None
    else:
        # for direct evaluation, we have the ground truth
        # but some steps can be None because ProcessBench dataset only includes
        # the annotation for the first error step
        y_true_instance_level = all(
            [y_true for y_true in y_true_step_level if y_true is not None]
        )

    if verification_score_type == "binary":
        # binary prediction
        # if all step-level predictions are "correct", we consider the instance-level prediction as "correct"
        # if there is no valid prediction, we consider the instance-level prediction as "correct" (i.e., None -> "correct")
        y_pred_instance_level = get_verification_score_from_binary_prediction(y_pred_step_level)
    elif "logprob" in verification_score_type:
        # remove None from y_pred_step_level
        # e.g., if <step_1>correct</step_1> is not found, y_pred_step_level[0] is None
        cleaned_y_pred_step_level = [
            y_pred for y_pred in y_pred_step_level if y_pred is not None
        ]
        
        # if all steps are missing, put a pseudo value
        # we don't want to use this response in best-of-k, so we put a very low value
        if len(cleaned_y_pred_step_level) == 0:
            cleaned_y_pred_step_level = [-1000000000.0]
        
        # take mean or min
        if verification_score_type == "logprob_min":
            y_pred_instance_level = np.min(cleaned_y_pred_step_level).item()
        else:
            raise ValueError(
                f"Unknown verification_score_type: {verification_score_type}"
            )
    else:
        raise ValueError(
            f"Unknown verification_score_type: {verification_score_type}"
        )
    
    # postprocessing for direct evaluation
    # if the instance-level prediction is None, we treat it as "correct"
    if "logprob" in verification_score_type:
        y_pred_step_level = [pred if pred is not None else 1000000000.0 for pred in y_pred_step_level]
    
    return {
        # step level
        "y_pred_step_level": y_pred_step_level,
        "is_y_pred_step_level_valid": is_y_pred_step_level_valid,
        "y_true_step_level": y_true_step_level,
        # instance level
        "y_pred_instance_level": y_pred_instance_level,
        "y_true_instance_level": y_true_instance_level,
        # check if all step scores are valid
        "is_all_step_scores_valid": is_all_step_scores_valid,
        "is_all_step_scores_invalid": is_all_step_scores_invalid,
        "invalid_steps_ratio": invalid_steps_ratio,
    }


def postprocess_prm_output(verification_score_type: str,
                           original_y_true: list[PRM_PRED],
                           evaluation_output: dict,
                           base_model_name: str,
                           logprobs: list[list[tuple[str, float]]] | None = None,
                           remove_step_if_y_true_is_none: bool=False) -> dict:
    """ Postprocess responses from reasoning LLM-based verifiers.
    LLM response format: reasoning <step_1>prediction</step_1>
    reasoning <step_2>prediction</step_2> ...
    (prediction: "correct" or "incorrect")
    
    Args:
        original_y_true: list of ground truth labels for each step
        evaluation_output: output from the verifier
        remove_step_if_y_true_is_none: Whether to skip steps with no ground 
            truth. For example, ProcessBench dataset includes None annotations
            for the steps after the first error step. In this case, we skip
            the steps after the first error step (set this argument as True).
            However, for sample-and-rank evaluation, we do not have the ground
            truth for each step, and we provide a pseudo original_y_true with 
            all None. In this case, we do not skip any steps (set this argument
            as False).
    """
    
    response: str = evaluation_output["response"]
    
    y_true_step_level: list[PRM_PRED] = []
    y_pred_step_level: list = []
    for step_idx in range(1, len(original_y_true) + 1):
        y_true = original_y_true[step_idx-1]
        if remove_step_if_y_true_is_none:
            if y_true is None:
                # skip steps with no ground truth
                # for example, processbench dataset only includes the annotation for the first error step
                # therefore, it does not include the annotation for the steps after the first error step
                # in this case, we skip the steps after the first error step (annotated as None)
                continue
        y_true_step_level.append(y_true)
        
        if verification_score_type == "binary":
            # binary prediction
            if f"<step_{step_idx}>correct</step_{step_idx}>" in response:
                y_pred_step_level.append(True)
            elif f"<step_{step_idx}>incorrect</step_{step_idx}>" in response:
                y_pred_step_level.append(False)
            else:
                y_pred_step_level.append(None)
        elif "logprob" in verification_score_type:
            if logprobs is None:
                raise ValueError("Log probabilities are not available.")
            
            y_pred_step_level.append(
                get_logprob_for_correct(
                    logprobs, step_idx, base_model_name=base_model_name
                )
            )
        else:
            raise ValueError(
                f"Unknown verification_score_type: {verification_score_type}"
            )
    
    return get_postprocessed_prm_output_format(
        y_pred_step_level=y_pred_step_level,
        y_true_step_level=y_true_step_level,
        verification_score_type=verification_score_type,
    )


# this is a new version
def postprocess_prm_output_from_vllm_reward_model(
        verification_output: dict, verification_model: str,
        verification_score_type: str,
        original_y_true: list[PRM_PRED] | None = None,
        remove_step_if_y_true_is_none: bool=False) -> dict:
    
    # get step level predictions
    if verification_model in []:
        # if future models need special treatment, add here
        raise NotImplementedError(
            f"Verification model {verification_model} is not supported yet."
        )
    elif verification_model == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
        # for this model, the reward scores are generated
        y_pred_step_level = verification_output["response"]
    else:
        classification_output = verification_output["response"]
        
        # the second score is for the "correct" class
        y_pred_step_level = [
            s[1] for s in classification_output
        ]
    
    # check length
    if original_y_true is not None:
        if len(original_y_true) != len(y_pred_step_level):
            raise ValueError(
                f"original_y_true and y_pred_step_level have different lengths: {len(original_y_true)} != {len(y_pred_step_level)}"
            )

    # invalid cases
    if original_y_true is not None and not remove_step_if_y_true_is_none:
        raise ValueError(
            "original_y_true is not None, but remove_step_if_y_true_is_none is False. " \
            "This should not happen."
        )
    
    # this is for downstream evaluation where we don't have the ground truth
    if original_y_true is None:
        # for downstream evaluation, we don't have the ground truth
        y_true_step_level = [None] * len(y_pred_step_level)
    else:
        y_true_step_level = original_y_true

    # get format
    postprocessed = get_postprocessed_prm_output_format(
        y_pred_step_level=y_pred_step_level,
        y_true_step_level=y_true_step_level,
        verification_score_type=verification_score_type,
    )

    # remove_step_if_y_true_is_none
    if remove_step_if_y_true_is_none:
        # this part is only used for direct evaluation
        if original_y_true is None:
            raise ValueError(
                "original_y_true should be provided when remove_step_if_y_true_is_none is True."
            )
        
        new_y_pred_step_level = []
        y_true_step_level = []
        
        for step_idx in range(len(original_y_true)):
            y_true = original_y_true[step_idx]
            if y_true is None:
                # skip steps with no ground truth
                # for example, processbench dataset only includes the annotation for the first error step
                # therefore, it does not include the annotation for the steps after the first error step
                # in this case, we skip the steps after the first error step (annotated as None)
                continue
            
            new_y_pred_step_level.append(y_pred_step_level[step_idx])
            y_true_step_level.append(y_true)
        y_pred_step_level = new_y_pred_step_level

        # update
        postprocessed["original_y_pred_step_level"] = postprocessed["y_pred_step_level"]
        postprocessed["original_y_true_step_level"] = postprocessed["y_true_step_level"]
        
        postprocessed["y_pred_step_level"] = y_pred_step_level
        postprocessed["y_true_step_level"] = y_true_step_level
    
    return postprocessed
