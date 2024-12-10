""" Load evaluation dataset """


import datasets


def load_existing_dataset(dataset_name: str, split: str = "test") -> datasets.Dataset:
    if dataset_name == "gsm8k":
        from src.load_dataset.dataset_loaders.\
            gsm8k import get_gsm8k_dataset
        return get_gsm8k_dataset(split=split)
    elif dataset_name == "math":
        from src.load_dataset.dataset_loaders.\
            math import get_math_dataset
        return get_math_dataset(split=split)
    elif dataset_name == "aqua":
        from src.load_dataset.dataset_loaders.\
            aqua import get_aqua_dataset
        return get_aqua_dataset(split=split)
    elif dataset_name == "orca_math":
        from src.load_dataset.dataset_loaders.\
            bigmath_math_word_problems import get_bigmath_orca_math
        return get_bigmath_orca_math(split=split)
    elif "bbh" in dataset_name:
        if split != "test":
            raise ValueError("bbh dataset only has test split.")
        
        from src.load_dataset.dataset_loaders.\
            bbh import get_bbh_dataset
        subset_name = dataset_name.split("_", 1)[1]
        return get_bbh_dataset(subset=subset_name)
    elif dataset_name == "folio":
        if split != "test":
            raise ValueError("folio dataset only has test split.")
        
        from src.load_dataset.dataset_loaders.\
            folio import get_folio_dataset
        # folio does not provide test split in public
        return get_folio_dataset(split="validation")
    elif dataset_name == "logicnli":
        if split != "test":
            raise ValueError("logicnli dataset only has test split.")

        from src.load_dataset.dataset_loaders.\
            logicnli import get_logicnli_dataset
        return get_logicnli_dataset(split="test")
    elif dataset_name == "anli":
        if split != "test":
            raise ValueError("we only use test set of anli.")

        from src.load_dataset.dataset_loaders.\
            anli import get_anli_dataset
        return get_anli_dataset(split="test")
    elif dataset_name == "hans":
        if split != "test":
            raise ValueError("we only use test set of split.")

        from src.load_dataset.dataset_loaders.\
            hans import get_hans_dataset
        return get_hans_dataset(split="test")
    elif dataset_name == "mmlu_pro_nomath":
        if split != "test":
            raise ValueError("mmlu_pro_nomath dataset only has test split.")
        
        from src.load_dataset.dataset_loaders.\
            mmlu_pro_nomath import get_mmlu_pro_nomath_dataset
        return get_mmlu_pro_nomath_dataset(split=split)
    elif dataset_name == "aime":
        if split != "test":
            raise ValueError("aime dataset only has test split.")
        
        from src.load_dataset.dataset_loaders.\
            aime import get_aime_dataset
        return get_aime_dataset(split=split)

    ###
    # only used for training
    elif dataset_name == "bigmath_math_word_problems":
        from src.load_dataset.dataset_loaders.\
            bigmath_math_word_problems import get_bigmath_math_word_problems_dataset
        return get_bigmath_math_word_problems_dataset(split=split)
    elif "metamathqa" in dataset_name:
        from src.load_dataset.dataset_loaders.\
            metamathqa import get_metamathqa_dataset
        # metamathqa only has train split
        return get_metamathqa_dataset(dataset_name, split=split)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
