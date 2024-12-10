from typing import Literal

import torch
import numpy as np
from transformers import AutoTokenizer


def get_step_token_position(
        tokenized_prompt: np.ndarray,
        tokenizer: AutoTokenizer,
        model_type: Literal["llama", "qwen"],
    ) -> np.ndarray:
    # we get the feature for the last token in the tag for the assistant
    # conversation. It includes the prediction for the next token (the
    # first token in the assistant's response), which is the reward
    step_token_position_position_candidates = []

    # we use multiple tokens to detect the target token id because
    # target tokens can be also included in other parts
    if model_type == "llama":
        # target token id
        target_token_id = tokenizer.encode(
            "\n\n", add_special_tokens=False)[0]
        step_token_position_position_candidates.append(
            tokenized_prompt == target_token_id
        )

        # assistant id
        assistant_token_id = tokenizer.encode(
            "assistant", add_special_tokens=False)[0]
        ids = np.where(
            tokenized_prompt == assistant_token_id
        )[0] + 2
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        mask[ids] = True
        step_token_position_position_candidates.append(mask)

        # <|end_header_id|>
        end_header_id = tokenizer.encode(
            "<|end_header_id|>", add_special_tokens=False)[0]
        ids = np.where(
            tokenized_prompt == end_header_id
        )[0] + 1
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        mask[ids] = True
        step_token_position_position_candidates.append(mask)

    elif model_type == "qwen":
        # target token id
        target_token_id = tokenizer.encode(
            "\n", add_special_tokens=False)[0]
        step_token_position_position_candidates.append(
            tokenized_prompt == target_token_id
        )

        # assistant id
        assistant_token_id = tokenizer.encode(
            "assistant", add_special_tokens=False)[0]
        ids = np.where(
            tokenized_prompt == assistant_token_id
        )[0] + 1
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        mask[ids] = True
        step_token_position_position_candidates.append(mask)

        # <|im_start|>
        end_header_id = tokenizer.encode(
            "<|im_start|>", add_special_tokens=False)[0]
        ids = np.where(
            tokenized_prompt == end_header_id
        )[0] + 2
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        mask[ids] = True
        step_token_position_position_candidates.append(mask)

    else:
        raise NotImplementedError(
            f"This code does not support {model_type} model"
        )

    # take and
    step_token_position = np.where(
        np.logical_and.reduce(step_token_position_position_candidates)
    )[0]

    return step_token_position


def extract_fover_scores(tokenized_prompt: np.ndarray,
        logits: torch.Tensor, tokenizer: AutoTokenizer) -> list[float]:
    
    model_type: Literal["llama", "qwen"] | None = None
    for model_type_candidate in ["llama", "qwen"]:
        if model_type_candidate in tokenizer.name_or_path.lower():
            model_type = model_type_candidate
            break
    if model_type is None:
        raise ValueError(
            f"Cannot find model type for {tokenizer.name_or_path}"
        )
    
    # get position of the step token ("correct" or "incorrect")
    step_token_position = get_step_token_position(
        tokenized_prompt=tokenized_prompt,
        tokenizer=tokenizer,
        model_type=model_type,
    )
    selected_logits = logits[step_token_position]

    # select the logits for the step token
    positive_token_id = tokenizer.encode(
        "correct", add_special_tokens=False)[0]
    negative_token_id = tokenizer.encode(
        "incorrect", add_special_tokens=False)[0]
    
    positive_logits = selected_logits[:, positive_token_id]
    negative_logits = selected_logits[:, negative_token_id]
    
    logits_pair = torch.stack([negative_logits, positive_logits], dim=1)
    scores = torch.nn.functional.softmax(logits_pair, dim=1)[:, 1].tolist()
    
    return scores
