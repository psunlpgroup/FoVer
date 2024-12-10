import re
import locale


# Set the locale for your program (e.g., for US formatting)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def string_to_int(s: str) -> int:
    """Convert a string to an integer."""
    
    return int(locale.atof(s))


def string_to_float(s: str) -> float:
    """Convert a string to a float. This is used for datasets like Orca-Math."""
    
    return float(locale.atof(s))


replace_list = ["%", "\\%"]

def normalize_gsm8k_final_answer(answer: str, cast: str="int") -> str:
    """Normalize the final answer for GSM8K."""
    
    for replaced_word in replace_list:
        answer = answer.replace(replaced_word, "")
    
    if cast == "int":
        return str(string_to_int(answer))
    elif cast == "float":
        return str(string_to_float(answer))
    else:
        raise ValueError(f"Unknown cast type: {cast}")
