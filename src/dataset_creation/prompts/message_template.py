def get_assistant_message(content: str, model_name: str):
    if "gemma" in model_name:
        return {"role": "model", "content": content}
    else:
        return {"role": "assistant", "content": content}

def get_user_message(content: str):
    return {"role": "user", "content": content}
