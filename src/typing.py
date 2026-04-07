from typing import Literal, TypedDict


# this is an old version of the dataset
TRAIN_DATA = Literal[
    "fldx2_symbol_with_cot",
    "isabelle_all_with_cot",
    "fldx2_symbol_with_cot,isabelle_all_with_cot",
]

# this is a new version of the dataset
TRAIN_DATA_MULTI_TURN = Literal[
    "FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_202512",
]

SPLIT = Literal["train", "validation", "test"]

BASE_MODEL = Literal[
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
OPTIMIZERS = Literal["AdamW", "RecAdam"]

PRM_PRED = Literal[True, False, None]

PROMPT_TYPE = Literal["few-shot", "zero-shot", "multi-turn"]

DOWNSTREAM_EVALUATION_MODE = Literal["model_selection", "final_evaluation"]

BOK_MODEL = Literal[
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


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
