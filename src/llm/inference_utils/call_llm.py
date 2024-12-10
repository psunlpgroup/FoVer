from typing import Optional, Union
import warnings

import torch
import PIL.Image
import vllm

from src.llm.inference_utils import InferenceModel
from src.llm.inference_utils.utils import LlmInferenceParamsTap, get_llm_input_dict, hf_model_names_list
from src.llm.inference_utils.cache_llm_response import read_cache, dump_cache

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

proprietary_models_list = ["gpt", "claude", "gemini"]


def update_cache_list_if_none(current_cache_list: list, new_cache_list: list) -> list:
    """ Updates the current cache list with the new cache list if the current cache is None. """
    
    updated_cache_list = []
    for current_cache, new_cache in zip(current_cache_list, new_cache_list):
        if current_cache is None:
            updated_cache_list.append(new_cache)
        else:
            updated_cache_list.append(current_cache)
    
    return updated_cache_list


def postprocess_vllm_logprob(logprobs: list[dict]) \
        -> list[list[tuple[str, float]]]:
    """ Postprocesses the logprobs from vLLM. """
    
    processed_logprobs_list = []
    for logprob in logprobs:
        logprobs_for_this_token = []
        for _, value in logprob.items():
            logprobs_for_this_token.append((value.decoded_token, value.logprob))
        processed_logprobs_list.append(logprobs_for_this_token)
    
    return processed_logprobs_list


def call_llm(model: InferenceModel, prompt: Union[str, list[str], list[dict], list[list[dict]]], params: LlmInferenceParamsTap,
             image: Optional[Union[PIL.Image.Image, list[PIL.Image.Image]]]=None, image_path: Optional[Union[str, list[str]]]=None,
             overwrite_cache: bool=False,
             ) -> Union[str, list[str]]:
    """ Calls the given LLM model. You may also provide images. """
    
    # if temperature is not zero and overwrite_cache is False, raise a warning
    if params.temperature > 0 and not overwrite_cache:
        warnings.warn("Temperature is not zero, but overwrite_cache is False." \
            " This lead to repeated responses is you use the same seed. " \
            f"The current seed is {params.seed}."
        )
    
    # this function is implemented for batch processing, but it also accepts single inputs
    # if the input is single, it will be converted to a list
    if type(prompt) == str:
        # single input
        prompt = [prompt]
    elif type(prompt) == list and type(prompt[0]) == dict:
        if model.model_name == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
            # in Skywork, one dict is a single input
            prompt = prompt
        else:
            # single input with conversation
            prompt = [prompt]
    
    batch_size = len(prompt)
    
    # when images are not provided
    if image is None:
        image = [None] * len(prompt)
    if image_path is None:
        image_path = [None] * len(prompt)
    
    input_dict_list, messages_list = [], []
    cache_list = []
    for p, img in zip(prompt, image):
        input_dict, parameters_dict = get_llm_input_dict(model.model_name, prompt=p, image=img, params=params, use_vllm=model.use_vllm is not None,)
        input_dict_list.append(input_dict)
        messages_list.append(input_dict["messages"])

        if overwrite_cache:  # do not use cache
            cache_list.append(None)
        else:  # read cache
            cache = read_cache(input_dict, image=None if any(m in model.model_name for m in proprietary_models_list) else img)
            cache_list.append(cache)
    
    # if any cache is None, we will update the cache
    save_cache = True if any(c is None for c in cache_list) else False
    if save_cache is True:
        model.load_model()
    
    # run the model
    responses_list = []

    ###
    # Proprietary models
    if any(m in model.model_name for m in proprietary_models_list):
        for idx in range(len(prompt)):
            cache = cache_list[idx]
            input_dict = input_dict_list[idx]
            message = messages_list[idx]
            
            if "gpt" in model.model_name:
                if cache is None:
                    import openai
                    
                    with open("../openai_api_key.txt", "r") as f:
                        api_key = f.read().strip()
                    
                    client = openai.OpenAI(api_key=api_key)
                    cache = client.chat.completions.create(**input_dict).dict()
                
                response = cache["choices"][0]["message"]["content"]
            
            elif "claude" in model.model_name:
                import anthropic
                
                if cache is None:
                    with open("../anthropic_api_key.txt", "r") as f:
                        api_key = f.read().strip()
                    
                    client = anthropic.Anthropic(api_key=api_key)
                    cache = client.messages.create(**input_dict).dict()
                
                response = cache["content"][0]["text"]
            
            elif "gemini" in model.model_name:
                if cache is None:
                    import google.generativeai as genai
                    with open("../google_api_key.txt", "r") as f:
                        api_key = f.read().strip()
                    genai.configure(api_key=api_key)
                    
                    api_model = genai.GenerativeModel(model.model_name)
                    cache = api_model.generate_content(input_dict["messages"]).text
                
                response = cache
            
            cache_list[idx] = cache
            responses_list.append(response)

    ###
    # Hugging Face models
    elif any(m in model.model_name for m in hf_model_names_list):
        if any(img is not None for img in image):
            raise NotImplementedError(f"Image input for {model.model_name} is not implemented yet")
        
        if save_cache:
            if model.use_vllm:  # use vLLM
                vllm_model: vllm.LLM = model.model
                assert vllm_model is not None, "vllm_model is None"
                
                if model.is_reward_model:
                    
                    if model.model_name == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
                        # https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
                        from openai import OpenAI
                        from transformers import AutoTokenizer
                        from model_utils.io_utils import prepare_input, derive_step_rewards_vllm
                        prm_model_path = model.model_name
                        tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)

                        processed_data = [prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n") for d in prompt]
                        input_ids, steps, reward_flags = zip(*processed_data)

                        openai_api_key = "EMPTY"
                        openai_api_base = "http://localhost:8081/v1"

                        try:
                            client = OpenAI(
                                # defaults to os.environ.get("OPENAI_API_KEY")
                                api_key=openai_api_key,
                                base_url=openai_api_base,
                            )
                            rewards = client.embeddings.create(
                                input=input_ids,
                                model=client.models.list().data[0].id,
                            )
                            step_rewards = derive_step_rewards_vllm(rewards, reward_flags)

                        except Exception as e:
                            import traceback

                            print()
                            print("--" * 20)
                            print("You need to start the vLLM server first.")
                            print("Please refer to https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B")
                            print("--" * 20)
                            print()

                            traceback.print_exc()
                            raise e
                        
                        output = step_rewards  # list[list[float]]
                    else:
                        # reward models
                        text_prompt = model.tokenizer.apply_chat_template(
                            prompt, tokenize=False
                        )
                        output = vllm_model.encode(text_prompt)
                else:
                    # standard models
                    sampling_params = vllm.SamplingParams(
                        temperature=params.temperature,
                        top_p=params.top_p,
                        top_k=params.top_k,
                        max_tokens=params.max_tokens,
                        seed=params.seed,
                        logprobs=20 if params.logprobs else None,
                    )
                    
                    output = vllm_model.chat(prompt, sampling_params)
                
                if model.is_reward_model:
                    processed_output = [
                        model.postprocess_reward_model_output(o) for o in output
                    ]
                elif params.logprobs:
                    processed_output = [
                        (
                            o.outputs[0].text,
                            postprocess_vllm_logprob(
                                o.outputs[0].logprobs
                            )
                        )
                        for o in output
                    ]
                else:
                    processed_output = [
                        o.outputs[0].text for o in output
                    ]
                
                cache_list = update_cache_list_if_none(
                    cache_list, processed_output
                )
            elif model.use_hf_pipeline is not None:  # use Hugging Face pipeline
                hf_pipeline = model.model
                assert hf_pipeline is not None, "hf_pipeline is None"
                
                eos_token_id = hf_pipeline.tokenizer.eos_token_id
                if type(eos_token_id) == list:
                    # some models have multiple eos tokens
                    eos_token_id = eos_token_id[0]
                
                hf_pipeline.tokenizer.pad_token_id = eos_token_id
                hf_pipeline.tokenizer.padding_side = "left"
                
                # we need to remove unused keys
                del parameters_dict["seed"]
                del parameters_dict["use_vllm"]
                
                with torch.no_grad():
                    cache_list = update_cache_list_if_none(
                        cache_list,
                        hf_pipeline(prompt, batch_size=batch_size, truncation=True, **parameters_dict)
                    )
            else:
                raise NotImplementedError(f"This code only supports vLLM and Hugging Face pipeline")
        
        # extract responses
        if model.use_vllm:
            # vLLM returns chat template when using vllm_model.generate
            # responses_list = [r.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "") for r in cache_list]
            # since we are using vllm_model.chat, we do not need to extract the chat template
            responses_list = cache_list
        elif model.use_hf_pipeline:
            responses_list = [cache[0]["generated_text"][-1]["content"] for cache in cache_list]
        else:
            raise NotImplementedError(f"This code only supports vLLM and Hugging Face pipeline")
    else:
        raise NotImplementedError(f"Model {model.model_name} is not supported")
    
    if save_cache:
        for idx in range(len(prompt)):
            cache = cache_list[idx]
            input_dict = input_dict_list[idx]
            img = image[idx]
            
            dump_cache(cache, input_dict, image=None if model.model_name in proprietary_models_list else img)
    
    if len(responses_list) == 1:
        return responses_list[0]
    else:
        return responses_list
