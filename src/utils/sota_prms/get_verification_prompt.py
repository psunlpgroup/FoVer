import transformers


def get_verification_prompt_for_sota_prms(
        problem: str, solution_steps: list[str], model_name: str
    ) -> list[dict] | dict:
    
    conversation: list[dict] | dict = []
    
    if model_name in [
                "Qwen/Qwen2.5-Math-7B-PRM800K",
                "Qwen/Qwen2.5-Math-PRM-7B",
            ]:
        tokenzier = transformers.AutoTokenizer.from_pretrained(model_name)
        
        content = "<extra_0>".join(solution_steps) + "<extra_0>"
        # truncate to 4096 - 512 tokens
        # 512 tokens are reserved for the input
        # I'm not sure why but we get error for very long inputs.
        # In our experiment, this error occurs only for one case.
        content_tokens = tokenzier(content)["input_ids"][:4096 - 512]
        content_truncated = tokenzier.decode(content_tokens)
        
        # refer to https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K
        conversation = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": content_truncated},
        ]
    elif model_name == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
        # Skywork uses line breaks as step tokens
        # we need to remove line breaks from the solution steps
        conversation = {
            "problem": problem,
            "response": "\n".join(
                [s.replace("\n", " ") for s in solution_steps]
            ),
        }
    elif model_name in [
                "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
                "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
            ]:
        conversation = []
        
        # https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm
        # https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm
        assistant_output = {"role": "assistant", "content": "+"}
        
        # first input should include the problem and the first step
        initial_input = "\n\n".join([problem, solution_steps[0]])
        conversation.append(
            {"role": "user", "content": initial_input}
        )
        conversation.append(assistant_output)
        
        # for each step after the first step, append the assistant output
        for idx in range(1, len(solution_steps)):
            conversation.append(
                {"role": "user", "content": solution_steps[idx]}
            )
            conversation.append(assistant_output)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return conversation

