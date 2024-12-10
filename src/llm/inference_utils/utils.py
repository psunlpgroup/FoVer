from typing import Optional
import base64

from tap import Tap
import PIL.Image


class LlmInferenceParamsTap(Tap):
    max_tokens: int = 256
    temperature: float = 0
    top_p: float = 0.9
    top_k: int = 40
    seed: int = 68
    logprobs: bool = False
    not_use_vllm_reward_task: bool = False


def encode_image_base64(image: PIL.Image.Image):
    """ Encodes the image as base64 """
    
    image.save("tmp", format=image.format)
    
    with open("tmp", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


hf_model_names_list = ["Llama-3", "gemma-2", "gemma-3", "Qwen2", "Qwen-2", "RLHFlow", "Skywork"]


def get_llm_input_dict(model: str, prompt: str, params: LlmInferenceParamsTap,
                       use_vllm: bool=False,
                       image: Optional[PIL.Image.Image]=None) -> tuple[dict, dict]:
    """ Returns the input dict for the given model
    
    Args:
        model (str): The model name
        prompt (str): The prompt
        params (LlmInferenceParamsTap): The parameters
        use_vllm (bool): Whether to use vLLM. Default is False.
        image (Optional[PIL.Image.Image]): The image input. Default is None.
    
    Returns:
        input_dict: The input dict
        parameters_dict: A sub-dict of input_dict that includes the parameters
    """
    
    message = get_llm_input(model, prompt, image)
    messages_dict = {
        "model": model,
        "messages": message,
    }
    
    # we add seed for all models
    # even when it is not used, it is useful when we generate multiple
    # responses with temperature > 0 and save the cache
    parameters_dict = {
        "seed": params.seed
    }
    
    # proprietary models
    proprietary_params = {
        "max_tokens": params.max_tokens, "temperature": params.temperature,
        "top_p": params.top_p,
    }
    if "gpt" in model:
        # parameters_dict.update({**proprietary_params, "seed": params.seed})
        parameters_dict.update(proprietary_params)
    elif any([m in model for m in ["claude", "gemini"]]):
        parameters_dict.update(proprietary_params)
    
    # huggingface models
    if any(hf_model_name in model for hf_model_name in hf_model_names_list):
        parameters_dict.update(
            {"max_new_tokens": params.max_tokens, "use_vllm": use_vllm}
        )
        
        if params.logprobs:
            parameters_dict.update({"logprobs": params.logprobs})
        
        if params.temperature == 0:
            parameters_dict.update({"do_sample": False, "temperature": None, "top_p": None})
        else:
            parameters_dict.update({"temperature": params.temperature, "top_p": params.top_p})
    
    # at least one parameter is added
    if len(parameters_dict) > 1:
        input_dict = {**messages_dict, **parameters_dict}
        return input_dict, parameters_dict
    
    # otherwise, the model is not supported
    raise NotImplementedError(f"Model {model} is not supported")


def get_llm_input(model_name: str, prompt: str, image: PIL.Image.Image):
    """ Returns the input (prompt or list of conversation) for the given model """
    
    if "gpt" in model_name or "claude" in model_name or "Qwen2-VL" in model_name:
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        if image is not None:
            base64_image = encode_image_base64(image)

            if "gpt" in model_name:
                content.insert(
                    0,
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image.format.lower()};base64,{base64_image}",
                            "detail": "high",
                        }
                    }
                )
            elif "claude" in model_name:
                content.insert(
                    0,
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image.format.lower()}",
                            "data": base64_image,
                        },
                    }
                )
            elif "Qwen2-VL" in model_name:
                content.insert(
                    0,
                    {
                        "type": "image",
                        "image": f"data:image/{image.format.lower()};base64,{base64_image}",
                    }
                )
        
        return [
            {
                "role": "user",
                "content": content
            }
        ]
    
    if "gemini" in model_name:
        if image is None:
            return prompt
        else:
            return [image, prompt]
    
    # huggingface models
    if any(hf_model_name in model_name for hf_model_name in hf_model_names_list):
        return prompt

    raise NotImplementedError(f"Model {model_name} is not supported")
