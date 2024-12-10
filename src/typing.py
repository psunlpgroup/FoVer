from typing import Literal, TypedDict


# this is an old version of the dataset
TRAIN_DATA = Literal[
    "fldx2_symbol_with_cot",
    "isabelle_all_with_cot",
    "fldx2_symbol_with_cot,isabelle_all_with_cot",
]

# this is a new version of the dataset
TRAIN_DATA_MULTI_TURN = Literal[
    "fldx2_symbol_multi_turn_balanced_last_step_20k",
    "isabelle_all_multi_turn_balanced_last_step_20k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k",
]

TRAIN_DATA_ABLATION = Literal[
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.50",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75",
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k",
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_duplicated_40k",
]


SPLIT = Literal["train", "validation", "test"]

BASE_MODEL = Literal[
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-27b-it",
]
OPTIMIZERS = Literal["AdamW", "RecAdam"]

PRM_PRED = Literal[True, False, None]

PROMPT_TYPE = Literal["few-shot", "zero-shot", "multi-turn"]

DOWNSTREAM_EVALUATION_MODE = Literal["model_selection", "final_evaluation"]


class ErrorLabelInstance(TypedDict):
    id: str
    problem: str
    proof_steps: list[str]
    all_process_correct: bool
    proof_step_correctness: list[bool]


class PrmDatasetInstance(TypedDict):
    """ Represents an instance in the PRM dataset.
    
    Attributes:
        id (str): The ID of the instance.
        messages (list[dict[str, str]]): The user and assistant messages.
        error_labels (list[PRM_PRED]): The error labels.
        base_dataset (BASE_DATA_NAME): The base dataset.
        proof_steps (list[str]): The proof steps.
    """
    
    id: str
    messages: list[dict[str, str]]  # user and assistant messages
    error_labels: list[PRM_PRED]
    base_dataset: str
    proof_steps: list[str]
