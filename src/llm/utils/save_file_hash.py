import hashlib
from typing import Union
from pathlib import Path


def get_md5_hash(file_path: str) -> str:
    """
    Calculate the MD5 hash of a file.

    :param file_path: Path to the file for which the MD5 hash is to be calculated.
    :return: MD5 hash string of the file's content.
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            # Read the file in chunks to handle large files efficiently
            for chunk in iter(lambda: file.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error: {e}"


def save_md5_hash(file_path: Union[str, Path]) -> None:
    """
    Calculate the MD5 hash of a file and save it to a file.

    :param file_path: Path to the file for which the MD5 hash is to be calculated.
    :param hash_file_path: Path to the file where the MD5 hash is to be saved.
    """
    file_path = str(file_path)
    
    md5_hash = get_md5_hash(file_path)
    hash_file_path = file_path + ".md5"
    with open(hash_file_path, 'w') as hash_file:
        hash_file.write(md5_hash)
