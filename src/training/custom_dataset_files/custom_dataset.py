# This code is based on https://github.com/meta-llama/llama-recipes/blob/48ba6805afa33a39332ed1874e630f6449204a99/src/llama_recipes/datasets/samsum_dataset.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import os
import datasets

import transformers

HF_ACCOUNT = os.getenv("HF_ACCOUNT")
if len(HF_ACCOUNT) == 0:
    raise ValueError("HF_ACCOUNT environment variable must be set")


def get_custom_dataset_for_all_models(dataset_config, tokenizer: transformers.PreTrainedTokenizer, split: str, model_name: str, dataset_name: str):
    model_short_name = model_name.split("/")[-1]
    dataset = datasets.load_dataset(f"{HF_ACCOUNT}/FoVer_{dataset_name}_{model_short_name}", split=split, trust_remote_code=True)
    
    def apply_prompt_template(sample):
        messages = sample["messages"]
        
        converted_messages = tokenizer.apply_chat_template(messages, tokenize=False)  # all messages
        
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)  # all but the last message
        # we do not want to include system prompt in the answer
        # so we do not directly apply tokenizer.apply_chat_template to the last message
        answer = converted_messages[len(prompt):]  # the last message
        
        return {
            "prompt": prompt,
            "answer": answer,
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    print(dataset[0])

    def tokenize_add_label(sample):
        # BOS and EOS tokens are added by tokenizer.apply_chat_template
        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"], add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    print(dataset[0])

    return dataset
