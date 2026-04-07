from pathlib import Path


fover_dataset_dir = Path("fover_dataset")
multitask_training_dataset_dir = Path("multitask_training_dataset")

direct_evaluation_datasets_dir = Path("direct_evaluation_datasets")


# dataset creation
intermediate_dir = Path("intermediate_outputs")
base_datasets_dir = intermediate_dir / "base_datasets"
error_labels_dir = intermediate_dir / "error_labels"
train_data_for_unsloth_dir = intermediate_dir / "train_data_for_unsloth"

model_inputs_dir = Path("model_inputs")
prompt_for_initial_generation_dir = model_inputs_dir / "prompt_for_initial_generation"
prompt_for_translation_dir = model_inputs_dir / "prompt_for_translation"

model_responses_dir = Path("model_responses")
dataset_creation_initial_answers_dir = model_responses_dir / "dataset_creation_initial_answers"
dataset_creation_translation_outputs_dir = model_responses_dir / "dataset_creation_translation_outputs"

# evaluation
performance_dir = Path("performance")
tables_dir = performance_dir / "tables"
performance_figures_dir = performance_dir / "figures"
downstream_evaluation_tables_dir = tables_dir / "downstream_evaluation"

manual_analysis_dir = Path("manual_analysis")

###
# fover dataset creation

def get_fover_dataset_path(dataset_name: str, model_name: str, split: str, suffix: str) -> Path:
    """ Get the path to the JSONL file of the dataset for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]

    if len(suffix) == 0:
        name = f"{split}.jsonl"
    else:
        name = f"{split}.{suffix}.jsonl"
    
    return fover_dataset_dir / dataset_name / model_short_name / name


def get_unsloth_training_dataset_path(dataset_name: str, model_name: str, split: str, suffix: str) -> Path:
    """Path to GRPO training data created for Unsloth runs."""
    model_short_name = model_name.split("/")[-1]

    if suffix:
        name = f"{split}.{suffix}.jsonl"
    else:
        name = f"{split}.jsonl"

    return train_data_for_unsloth_dir / dataset_name / model_short_name / name


def get_base_dataset_path(dataset_name: str, split: str) -> Path:
    """ Get the path to the JSONL file of the base dataset for the given dataset and split. """
    return base_datasets_dir / dataset_name / f"{split}.jsonl"


def get_prompt_for_initial_generation_path(dataset_name: str, model_name: str, split: str, seed: int | str) -> Path:
    """ Get the path to the JSONL file of the prompts for initial generation for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    return prompt_for_initial_generation_dir / dataset_name / model_short_name / f"seed={seed}" / f"{split}.jsonl"


def get_initial_answers_path(dataset_name: str, model_name: str, split: str, seed: int | str) -> Path:
    """ Get the path to the JSONL file of the initial answers file for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    return dataset_creation_initial_answers_dir / dataset_name / model_short_name / f"seed={seed}" / f"{split}.jsonl"


def get_error_labels_path(dataset_name: str, model_name: str, split: str, seed: int | str) -> Path:
    """ Get the path to the JSONL file of the error labels file for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    return error_labels_dir / dataset_name / model_short_name / f"seed={seed}" / f"{split}.jsonl"


def get_prompt_for_translation_path(dataset_name: str, generation_model_name: str, translation_model_name: str, split: str, field: str) -> Path:
    """ Get the path to the JSONL file of the prompts for symbol-to-text translation for the given dataset, model, split, and field.
    
    Args:
        field: Either 'problem' or 'proof_steps' to distinguish translation targets.
    """
    generation_model_short_name = generation_model_name.split("/")[-1]
    translation_model_short_name = translation_model_name.split("/")[-1]
    
    return prompt_for_translation_dir / dataset_name / f"translation={translation_model_short_name}" / generation_model_short_name / field / f"{split}.jsonl"


def get_translation_outputs_path(dataset_name: str, generation_model_name: str, translation_model_name: str, split: str, field: str) -> Path:
    """ Get the path to the JSONL file of the translation outputs for the given dataset, model, split, and field.
    
    Args:
        field: Either 'problem' or 'proof_steps' to distinguish translation targets.
    """
    generation_model_short_name = generation_model_name.split("/")[-1]
    translation_model_short_name = translation_model_name.split("/")[-1]
    
    return dataset_creation_translation_outputs_dir / dataset_name / f"translation={translation_model_short_name}" / generation_model_short_name / field / f"{split}.jsonl"


# multitask training dataset
def get_multitask_training_dataset_path(dataset_name: str, model_name: str, split: str) -> Path:
    """ Get the path to the JSONL file of the dataset for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    return multitask_training_dataset_dir / dataset_name / model_short_name / f"{split}.jsonl"


###
# direct evaluation

def get_short_fover_dataset_name(dataset_name: str) -> str:
    """ Get the short name of the fover dataset for the given dataset name.
    The dataset name here is the path to the local dataset. """
    dataset_name_split = dataset_name.split("/")
    return dataset_name_split[-2] + "-" + dataset_name_split[-1]


def get_direct_evaluation_dataset_path(dataset_name: str, split: str) -> Path:
    """ Get the path to the JSONL file of the evaluation dataset for the given dataset and split. """
    return direct_evaluation_datasets_dir / dataset_name / f"{split}.jsonl"


def get_prompt_for_direct_evaluation_path(dataset_name: str, model_name: str, split: str, prompt_type: str) -> Path:
    """ Get the path to the JSONL file of the prompts for the direct evaluation for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = get_short_fover_dataset_name(dataset_name) if "fover" in dataset_name else dataset_name.split("/")[-1]
    return model_inputs_dir / "prompt_for_direct_evaluation" / dataset_short_name / prompt_type / model_short_name / f"{split}.jsonl"


def get_direct_evaluation_outputs_path(dataset_name: str, model_name: str, split: str, prompt_type: str) -> Path:
    """ Get the path to the JSONL file of the direct evaluation output for the given dataset, model, split, and prompt type. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = get_short_fover_dataset_name(dataset_name) if "fover" in dataset_name else dataset_name.split("/")[-1]
    return model_responses_dir / "direct_evaluation_outputs" / dataset_short_name / prompt_type / model_short_name / f"{split}.jsonl"


def get_direct_evaluation_metrics_path(dataset_name: str, verification_model_name: str, split: str, verification_prompt_type: str) -> Path:
    """ Get the path to the JSONL file of the performance metrics for the direct evaluation (evaluation on the error detectiont task). """
    verification_model_short_name = verification_model_name.split("/")[-1]
    dataset_short_name = get_short_fover_dataset_name(dataset_name) if "fover" in dataset_name else dataset_name.split("/")[-1]
    return performance_dir / "evaluation_metrics" / "direct_evaluation" / dataset_short_name / verification_prompt_type / f"verification={verification_model_short_name}" / f"{split}.json"


###
# downstream evaluation
downstream_evaluation_model_inputs_dir = model_inputs_dir / "downstream_evaluation"
downstream_evaluation_model_responses_dir = model_responses_dir / "downstream_evaluation"

def get_prompt_for_downstream_evaluation_initial_responses_path(dataset_name: str, model_name: str, split: str, prompt_type: str, updated_save_directory: str | None = None) -> Path:
    """ Get the path to the JSONL file of the prompts for the downstream evaluation for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    
    if updated_save_directory is None:
        # default directory
        base_dir = downstream_evaluation_model_inputs_dir / "prompt_for_initial_responses"
    else:
        # when an updated save directory is provided
        # e.g., generating initial responses for training data creation
        base_dir = Path(updated_save_directory)
    
    return base_dir / dataset_short_name / prompt_type / model_short_name / f"{split}.jsonl"


def get_downstream_evaluation_initial_responses_path(dataset_name: str, model_name: str, split: str, prompt_type: str, sample_idx: int, updated_save_directory: str | None = None) -> Path:
    """ Get the path to the JSONL file of the initial responses for the downstream evaluation for the given dataset, model, split, prompt type, and sample_idx. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    
    if updated_save_directory is None:
        # default directory
        base_dir = downstream_evaluation_model_responses_dir / "downstream_evaluation_initial_responses"
    else:
        # when an updated save directory is provided
        # e.g., generating initial responses for training data creation
        base_dir = Path(updated_save_directory)
    return base_dir / dataset_short_name / prompt_type / model_short_name / f"{sample_idx}" / f"{split}.jsonl"


def get_prompt_for_extracting_answers_from_downstream_evaluation_initial_responses_path(dataset_name: str, model_name: str, split: str, sample_idx: int, updated_save_directory: str | None = None) -> Path:
    """ Get the path to the JSONL file of the prompts for extracting answers from the downstream evaluation initial responses for the given dataset, model, and split. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    
    if updated_save_directory is None:
        # default directory
        base_dir = downstream_evaluation_model_inputs_dir / "prompt_for_extracting_answers_from_initial_responses"
    else:
        # when an updated save directory is provided
        # e.g., generating initial responses for training data creation
        base_dir = Path(updated_save_directory)
    
    return base_dir / dataset_short_name / model_short_name / f"{sample_idx}" / f"{split}.jsonl"


def get_prompt_for_verification_for_sample_and_rank_path(dataset_name: str, model_name: str, split: str, prompt_type: str, few_shot_verification: bool, sample_idx: int) -> Path:
    """ Get the path to the JSONL file of the prompts for verification of sample-and-rank for the given dataset, model, split, prompt type, and sample_idx. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]

    full_prompt_type = f"{prompt_type}_few-shot" if few_shot_verification else prompt_type

    return downstream_evaluation_model_inputs_dir / "prompt_for_verification_of_sample_and_rank" / dataset_short_name / full_prompt_type / model_short_name / f"{sample_idx}" / f"{split}.jsonl"


def get_prompt_for_verification_for_sample_and_rank_by_sota_prms_path(dataset_name: str, model_name: str, split: str, prompt_type: str, sample_idx: int, verification_model_name: str) -> Path:
    """ Get the path to the JSONL file of the prompts for verification of sample-and-rank by baseline models for the given dataset, model, split, prompt type, and sample_idx. """
    model_short_name = model_name.split("/")[-1]
    verification_model_short_name = verification_model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    return downstream_evaluation_model_inputs_dir / "prompt_for_verification_of_sample_and_rank_by_sota_prms" / dataset_short_name / prompt_type / model_short_name / f"verification_model={verification_model_short_name}" / f"{sample_idx}" / f"{split}.jsonl"


def get_verification_for_sample_and_rank_outputs_path(dataset_name: str, initial_response_model_name: str, verification_model_name: str, split: str, prompt_type: str, few_shot_verification: bool, sample_idx: int) -> Path:
    """ Get the path to the JSONL file of the verification output for sample-and-rank for the given dataset, model, split, prompt type, and sample_idx. """
    initial_response_model_short_name = initial_response_model_name.split("/")[-1]
    verification_model_short_name = verification_model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]

    updated_prompt_type = prompt_type if not few_shot_verification else f"{prompt_type}_few-shot"

    return downstream_evaluation_model_responses_dir / "verification_for_sample_and_rank_outputs" / dataset_short_name / updated_prompt_type / f"initial_generation={initial_response_model_short_name}" / f"{sample_idx}" / f"verification={verification_model_short_name}" / f"{split}.jsonl"


def get_verification_scores_for_sample_and_rank_path(dataset_name: str, base_model_name: str, verification_model_name: str, verification_score_type: str, split: str, prompt_type: str, few_shot_verification: bool) -> Path:
    """ Get the path to the JSONL file of the verification scores for sample-and-rank for the given dataset, base model, verification model, split, and prompt type. """
    base_model_short_name = base_model_name.split("/")[-1]
    verification_model_short_name = verification_model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    full_prompt_type = f"{prompt_type}_few-shot" if few_shot_verification else prompt_type
    return downstream_evaluation_model_responses_dir / "verification_scores_for_sample_and_rank" / dataset_short_name / full_prompt_type / f"base_model={base_model_short_name}" / f"verification_model={verification_model_short_name}" / f"verification_score_type={verification_score_type}" / f"{split}.jsonl"


def get_best_sample_and_rank_output_path(dataset_name: str, base_model_name: str, verification_model_name: str, verification_prompt_type: str, few_shot_verification: bool, verification_score_type: str, selection_method: str, sample_k: int, split: str) -> Path:
    """ Get the path to the JSONL file of the best sample-and-rank output for the given dataset, base model, verification model, split, and prompt type. """
    base_model_short_name = base_model_name.split("/")[-1]
    verification_model_short_name = verification_model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    full_verification_prompt_type = f"{verification_prompt_type}_few-shot" if few_shot_verification else verification_prompt_type
    return downstream_evaluation_model_responses_dir / "best_sample_and_rank_output" / dataset_short_name / full_verification_prompt_type / f"base_model={base_model_short_name}" / f"verification_model={verification_model_short_name}" / f"verification_score_type={verification_score_type}" / f"selection_method={selection_method}" / f"sample_k={sample_k}" / f"{split}.jsonl"


def get_downstream_evaluation_metrics_path(dataset_name: str, model_name: str, prediction_path: str | Path, split: str="test") -> Path:
    """ Get the path to the JSON file of the performance metrics for the downstream evaluation (best-of-k). """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    processed_prediction_path = "/".join(str(prediction_path).split("/")[2:-1])
    return performance_dir / "evaluation_metrics" / "downstream_evaluation" / dataset_short_name / model_short_name / processed_prediction_path / f"{split}.json"


# self-consistency
def get_majority_vote_output_path(dataset_name: str, model_name: str, split: str, prompt_type: str) -> Path:
    """ Get the path to the JSONL file of the majority vote output for the given dataset, model, split, and prompt type. """
    model_short_name = model_name.split("/")[-1]
    dataset_short_name = dataset_name.split("/")[-1]
    return downstream_evaluation_model_responses_dir / "majority_vote_output" / dataset_short_name / prompt_type / model_short_name / f"{split}.jsonl"


def get_prm_guided_search_output_path(
    dataset_name: str,
    generator_model_name: str,
    prm_name: str,
    split: str,
    beam_size: int,
    per_beam_samples: int,
    max_steps: int,
) -> Path:
    """Get the path to the JSONL output file for PRM-guided search."""
    generator_model_short_name = generator_model_name.split("/")[-1]
    prm_short_name = prm_name.split("/")[-1]
    dataset_short_name = get_short_fover_dataset_name(dataset_name) if "fover" in dataset_name else dataset_name.split("/")[-1]

    return model_responses_dir / "prm_guided_search_outputs" / dataset_short_name / \
        f"generator={generator_model_short_name}" / f"prm={prm_short_name}" / \
        f"beam_size={beam_size}" / f"per_beam_samples={per_beam_samples}" / \
        f"max_steps={max_steps}" / f"{split}.jsonl"


# manual analysis
def get_annotation_csv_path(
    model_name: str,
    train_data_name: str,
    dataset_name: str,
    case_type: str,
    annotated: bool = False,
) -> Path:
    """
    Get the path to the annotation csv file for a specific model, train data,
    dataset, and case type.
    """

    directory_name = "annotated_csv" if annotated else "annotation_csv"

    model_short_name = model_name.split("/")[-1]
    annotation_csv_path = manual_analysis_dir / directory_name / \
        f"{model_short_name}_{train_data_name}_" \
        f"{dataset_name}_{case_type}.csv"
    return annotation_csv_path


# training
finetuned_models_dir = Path("finetuned_models")
