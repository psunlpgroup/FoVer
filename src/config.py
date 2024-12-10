import os
from typing import Union

from src.typing import TRAIN_DATA, TRAIN_DATA_MULTI_TURN, TRAIN_DATA_ABLATION, \
    SPLIT, BASE_MODEL, OPTIMIZERS, DOWNSTREAM_EVALUATION_MODE


###
# training datasets
splits_list: tuple[SPLIT] = (SPLIT.__args__)

# base datasets
train_dataset_names_list: list[TRAIN_DATA] = list(TRAIN_DATA.__args__)
train_dataset_names_list_multi_turn: list[TRAIN_DATA_MULTI_TURN] = list(
    TRAIN_DATA_MULTI_TURN.__args__
)

full_names_dict = {
    "fldx2_symbol": "hitachi-nlp/FLDx2",
}
base_datasets_display_name_dict: dict[Union[TRAIN_DATA, TRAIN_DATA_MULTI_TURN, TRAIN_DATA_ABLATION], str] = {
    "fldx2_symbol_multi_turn_balanced_last_step_20k": "FLDx2 (symbol) balanced last step 20k",
    "isabelle_all_multi_turn_balanced_last_step_20k": "Isabelle balanced last step 20k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k": "FLDx2 (symbol) + Isabelle balanced last step 40k",
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25": "FLDx2 (symbol) + Isabelle balanced last step 20k (correct=0.25)",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.50": "FLDx2 (symbol) + Isabelle balanced last step 20k (correct=0.50)",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75": "FLDx2 (symbol) + Isabelle balanced last step 20k (correct=0.75)",
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k": "FLDx2 (symbol) + Isabelle balanced last step 5k (duplicated=40k)",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k": "FLDx2 (symbol) + Isabelle balanced last step 10k (duplicated=40k)",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k": "FLDx2 (symbol) + Isabelle balanced last step 20k (duplicated=40k)",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_duplicated_40k": "FLDx2 (symbol) + Isabelle balanced last step 40k (duplicated=40k)",
}

base_model_names: list[BASE_MODEL] = list(BASE_MODEL.__args__)
base_model_names = [
    name for name in base_model_names if name not in ["google/gemma-3-27b-it"]
]


dataset_name_to_hf_name_dict = {
    "fldx2_symbol_multi_turn_10k": "FormalLogic_10k",
    "isabelle_all_multi_turn_10k": "FormalProof_10k",
    "fldx2_symbol-isabelle_all_multi_turn_10k": "PRM_FormalLogic-FormalProof_10k",
    #
    "fldx2_symbol_multi_turn_balanced_last_step_20k": "PRM_FormalLogic_balanced_last_step_20k",
    "isabelle_all_multi_turn_balanced_last_step_20k": "PRM_FormalProof_balanced_last_step_20k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k": "FormalLogic-FormalProof_LastStepBalanced_20k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k": "FormalLogic-FormalProof_LastStepBalanced_40k",
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25": "PRM_FormalLogic-FormalProof_balanced_last_step_20k_correct=0.25",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75": "PRM_FormalLogic-FormalProof_balanced_last_step_20k_correct=0.75",
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k": "PRM_FormalLogic-FormalProof_balanced_last_step_5k_duplicated=40k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k": "PRM_FormalLogic-FormalProof_balanced_last_step_10k_duplicated=40k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k": "PRM_FormalLogic-FormalProof_balanced_last_step_20k_duplicated=40k",
    #
    "fldx2_symbol_multi_turn": "PRM_FormalLogic",
    "isabelle_all_multi_turn": "PRM_FormalProof",
    "fldx2_symbol_multi_turn,isabelle_all_multi_turn": "PRM_FormalLogic-FormalProof",
}

# this function is used to get key for llama-factory
def get_fover_dataset_name(base_model_name: str, base_dataset_name: str) -> str:
    """ Get a pretty dataset name for the given base model and dataset names. """
    
    # Validate the base model name
    if base_dataset_name not in dataset_name_to_hf_name_dict.keys():
        raise ValueError(f"Invalid dataset name: {base_dataset_name}")
    hf_dataset_name = dataset_name_to_hf_name_dict.get(base_dataset_name)
    
    # Validate the model name
    if base_model_name not in base_model_names:
        raise ValueError(f"Invalid model name: {base_model_name}")
    short_model_name = base_model_name.split("/")[-1]
    
    # HF_ACCOUNT environment variable must be set
    hf_account = os.getenv("HF_ACCOUNT")
    if hf_account is None:
        raise ValueError("Please set your HuggingFace to HF_ACCOUNT environment variable: export HF_ACCOUNT=your_username")
    
    return f"{hf_account}/FoVer_{hf_dataset_name}_{short_model_name}"

###
# evaluation datasets
import os
from src.path import direct_evaluation_datasets_dir

HF_ACCOUNT = os.getenv("HF_ACCOUNT")
if HF_ACCOUNT is None or len(HF_ACCOUNT) == 0:
    raise ValueError("Please set your HuggingFace to HF_ACCOUNT environment variable: export HF_ACCOUNT=your_username")

# direct evaluation
from src.path import get_fover_dataset_path
processbench_splits = ["gsm8k", "math", "olympiadbench", "omnimath"]
def get_direct_evaluation_datasets_list(base_model_name: str, train_data_name: str | None, verification_prompt_type: str="zero-shot") -> list[str]:
    direct_evaluation_datasets_list = []
    
    # add processbench datasets
    for processbench_split in processbench_splits:
        direct_evaluation_datasets_list.append(str(direct_evaluation_datasets_dir / f"processbench_{processbench_split}"))
    
    # add fover datasets
    if train_data_name is None:
        # This is a base model. Evaluate on all training datasets.
        train_data_names_list_ = {
            "zero-shot": train_dataset_names_list,
            "multi-turn": [
                d for d in train_dataset_names_list_multi_turn
                if "balanced" not in d
            ],
        }[verification_prompt_type]
    
    else:
        # This is a fine-tuned model. Evaluated only on the training dataset.
        if "balanced" in train_data_name:
            # balanced dataset is only for training
            # we use the original dataset for evaluation
            if "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step" in train_data_name:
                evaluation_train_data_name = "fldx2_symbol-isabelle_all_multi_turn_10k"
            elif "fldx2_symbol_multi_turn_balanced_last_step" in train_data_name:
                evaluation_train_data_name = "fldx2_symbol_multi_turn_10k"
            elif "isabelle_all_multi_turn_balanced_last_step" in train_data_name:
                evaluation_train_data_name = "isabelle_all_multi_turn_10k"
            else:
                raise ValueError(f"Invalid training dataset name: {train_data_name}")
        else:
            evaluation_train_data_name = train_data_name
        
        train_data_names_list_ = [evaluation_train_data_name]
    
    for train_data_name_ in train_data_names_list_:
        direct_evaluation_datasets_list.append(
            str(get_fover_dataset_path(
                dataset_name=train_data_name_,
                model_name=base_model_name, split=""
            ).parent)
        )
    
    return direct_evaluation_datasets_list


direct_evaluation_dataset_pretty_name_dict = {
    "processbench_gsm8k": "GSM8K",
    "processbench_math": "MATH",
    "processbench_olympiadbench": "OlympiadBench",
    "processbench_omnimath": "OmniMath",
}

def get_direct_evaluation_dataset_pretty_name(dataset_name: str) -> str:
    dataset_short_name = dataset_name.split("/")[-1]
    
    if dataset_short_name in direct_evaluation_dataset_pretty_name_dict.keys():
        return direct_evaluation_dataset_pretty_name_dict[dataset_short_name]
    else:
        # FoVer dataset
        for key, value in base_datasets_display_name_dict.items():
            dataset_short_name = dataset_short_name.replace(key, value)
        for key, value in model_display_name_dict.items():
            model_short_name = key.split("/")[-1]
            dataset_short_name = dataset_short_name.replace(
                model_short_name, value
            )
        return dataset_short_name

# downstream evaluation
downstream_evaluation_for_model_selection_datasets_list = [
    "orca_math", "bbh_logical_deduction_three_objects",
    "bbh_boolean_expressions",
]
downstream_evaluation_datasets_list = [
    "gsm8k", "math", "aqua", "aime",
    "folio", "logicnli",
    "anli", "hans",
    "mmlu_pro_nomath",
    "bbh_temporal_sequences", "bbh_tracking_shuffled_objects_three_objects",
    "bbh_word_sorting",
]
def get_downstream_evaluation_datasets_lsit(
        evaluation_mode: DOWNSTREAM_EVALUATION_MODE
    ) -> list[str]:
    if evaluation_mode == "model_selection":
        return downstream_evaluation_for_model_selection_datasets_list
    elif evaluation_mode == "final_evaluation":
        return downstream_evaluation_datasets_list
    else:
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")


display_name_of_downstream_evaluation_dataset_dict = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "aqua": "AQuA-RAT",
    "aime": "AIME",
    "orca_math": "Orca-Math",
    "folio": "FOLIO",
    "logicnli": "LogicNLI",
    "anli": "ANLI",
    "hans": "HANS",
    "mmlu_pro_nomath": "MMLU-Pro-NoMath",
    "bbh_logical_deduction_three_objects": "BBH Logic",
    "bbh_formal_fallacies": "Fallacies",
    "bbh_temporal_sequences": "Temporal",
    "bbh_tracking_shuffled_objects_three_objects": "Tracking",
    "bbh_word_sorting": "Sorting",
    "bbh_boolean_expressions": "Boolean",
}
def get_downstream_evaluation_datasets_display_name_list(
        evaluation_mode: DOWNSTREAM_EVALUATION_MODE
    ) -> list[str]:
    dataset_names_list = get_downstream_evaluation_datasets_lsit(
        evaluation_mode
    )
    return [
        display_name_of_downstream_evaluation_dataset_dict[dataset_name]
        for dataset_name in dataset_names_list
    ]

downstream_evaluation_dataset_name_to_category_dict = {
    "gsm8k": "math",
    "math": "math",
    "aqua": "math",
    "aime": "math",
    "orca_math": "math",
    "folio": "logic",
    "logicnli": "logic",
    "anli": "nli",
    "hans": "nli",
    "mmlu_pro_nomath": "mmlu",
    "bbh_logical_deduction_three_objects": "bbh",
    "bbh_formal_fallacies": "bbh",
    "bbh_temporal_sequences": "bbh",
    "bbh_tracking_shuffled_objects_three_objects": "bbh",
    "bbh_word_sorting": "bbh",
    "bbh_boolean_expressions": "bbh",
}


###
# models
sota_prms_dict: dict[BASE_MODEL, list[str]] = {
    "meta-llama/Llama-3.1-8B-Instruct": [
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        "Qwen/Qwen2.5-Math-7B-PRM800K",
        "Qwen/Qwen2.5-Math-PRM-7B",
        "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
    ]
}
sota_prms_list = [verifier for verifiers in sota_prms_dict.values()
                  for verifier in verifiers]


optimizers_list = OPTIMIZERS.__args__

finetuned_verification_models_dict: dict[BASE_MODEL, dict[str, dict[Union[TRAIN_DATA, TRAIN_DATA_MULTI_TURN, TRAIN_DATA_ABLATION], list[str]]]] = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "AdamW": {
            "fldx2_symbol_multi_turn_balanced_last_step_20k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-6_0426",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_2.0e-6_0426",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_5.0e-6_0426",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-5_0426",
            ],
            "isabelle_all_multi_turn_balanced_last_step_20k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-6_0429",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_2.0e-6_0429",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_5.0e-6_0429",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-5_0429",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-6_0430",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6_0430",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_5.0e-6_0430",
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-5_0430",
            ],
            #
            #
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k_1.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k_1.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k_1.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_duplicated_40k": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-6_0430",
            ],
            #
            #
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25_1.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.50": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_1.0e-6_0429",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75": [
                "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75_1.0e-6",
            ],
        }
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "AdamW": {
            "fldx2_symbol_multi_turn_balanced_last_step_20k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-6_0427",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_2.0e-6_0427",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_5.0e-6_0427",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-5_0427",
            ],
            "isabelle_all_multi_turn_balanced_last_step_20k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-6_0429",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_2.0e-6_0429",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_5.0e-6_0429",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-5_0429",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-6_0430",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6_0430",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_5.0e-6_0430",
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-5_0430",
            ],
            #
            #
            #
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k_2.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k_2.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k_2.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_duplicated_40k": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-6_0430",
            ],
            #
            #
            #
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25_2.0e-6",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.50": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_1.0e-6_0429",
            ],
            "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75": [
                "llama_factory_finetuned_models/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75_2.0e-6",
            ],
        },
    },
}


model_display_name_dict = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen/Qwen2-7B-Instruct": "Qwen2 7B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
    "google/gemma-2-9b-it": "Gemma 2 9B",
    "google/gemma-3-27b-it": "Gemma 3 27B",
}


# update model_config_dict
model_config_dict = {}
for base_model in base_model_names:
    for optimizer in finetuned_verification_models_dict[base_model].keys():
        for dataset_name in train_dataset_names_list \
                + train_dataset_names_list_multi_turn:
            try:
                model_path_list = finetuned_verification_models_dict[
                    base_model][optimizer][dataset_name]
            except KeyError:
                continue
            
            base_dataset_display_name = \
                base_datasets_display_name_dict[dataset_name]
            
            for model_path in model_path_list:
                model_config_dict[model_path] = {
                    "finetuning": f"FoVer-{base_dataset_display_name}",
                    "optimizer": optimizer
                }
