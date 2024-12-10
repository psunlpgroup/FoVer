""" Cache outputs from LLMs """

from typing import Optional, Union
from pathlib import Path
import json
import hashlib

import PIL.Image

from src.llm.inference_utils.utils import encode_image_base64


default_cache_path = Path("./llm_cache")


class FileEncoder(json.JSONEncoder):
    """ JSON encoder that encodes PIL images as base64 to avoid serialization issues """
    
    def default(self, obj):
        if isinstance(obj, PIL.Image.Image):
            return encode_image_base64(obj)
        return super().default(obj)


def dict_to_cache_path(input_dict: dict, cache_dir: Union[str, Path] = default_cache_path):
    """ Returns the cache path for the given input_dict """
    
    # use FileEncoder to encode PIL images as base64 to avoid serialization issues
    str_dict = json.dumps(input_dict, sort_keys=True, cls=FileEncoder)
    
    hash_object = hashlib.sha512(str_dict.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    
    subdir = hex_dig[:2]
    file_name = hex_dig[2:]
    
    return Path(cache_dir) / subdir / f"{file_name}.json"


def add_image_hash_to_input_dict(input_dict: dict, image: Optional[PIL.Image.Image]):
    """ Adds the image hash to the input_dict when an image is provided """
    
    if image is None:
        return input_dict
    
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return {**input_dict, "image_hash": image_hash}


def read_cache(input_dict: dict, image: Optional[PIL.Image.Image] = None, cache_dir: Union[str, Path] = default_cache_path):
    """ Reads the cache for the given input_dict """
    
    input_dict = add_image_hash_to_input_dict(input_dict, image)
    cache_path = dict_to_cache_path(input_dict, cache_dir)
    
    if not Path(cache_path).exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading cache {cache_path}")
        raise e.with_traceback(e.__traceback__)


def dump_cache(cache: dict, input_dict: dict, image: PIL.Image.Image, cache_dir: Union[str, Path] = default_cache_path):
    """ Dumps the cache for the given input_dict """
    
    input_dict = add_image_hash_to_input_dict(input_dict, image)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dict_to_cache_path(input_dict, cache_dir)
    
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=4)
