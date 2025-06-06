import transformers

from src.training.custom_dataset_files.custom_dataset import get_custom_dataset_for_all_models


def get_custom_dataset(dataset_config, tokenizer: transformers.PreTrainedTokenizer, split: str):
    return get_custom_dataset_for_all_models(dataset_config, tokenizer, split, "Qwen/Qwen2-7B-Instruct", "fldx2_symbol")
