from collections import Counter

from src.typing import PrmDatasetInstance, PRM_PRED


def get_basic_dataset_stats(instance: list) -> dict:
    return {
        "num_instances": len(instance),
    }


def get_instance_label(instance: PrmDatasetInstance) -> bool:
    """ Get the instance label.
    
    Args:
        instance (PrmDatasetInstance): The instance.
    
    Returns:
        output (bool): The instance label.
    """
    
    if False in instance["error_labels"]:
        return False
    else:
        # None is considered as True
        return True


def get_prm_dataset_stats(dataset: list[PrmDatasetInstance]) -> dict:
    """ Get the statistics of the dataset.
    
    Args:
        dataset (list[PrmDatasetInstance]): The dataset.
    
    Returns:
        output (dict): The statistics of the dataset.
    """
    
    stat = {"instance_level": {}, "step_level": {}}
    
    # step level
    all_error_labels_list = []
    for instance in dataset:
        all_error_labels_list.extend(instance["error_labels"])
    
    stat["step_level"]["num_steps"] = len(all_error_labels_list)
    step_level_counter = Counter(all_error_labels_list)
    stat["step_level"]["label_count"] = dict(step_level_counter)
    stat["step_level"]["label_ratio"] = {
        k: v / len(all_error_labels_list) for k, v in step_level_counter.items()
    }

    # last step
    last_step_labels = [instance["error_labels"][-1] for instance in dataset]
    last_step_counter = Counter(last_step_labels)
    stat["step_level"]["last_step_label_count"] = dict(last_step_counter)
    stat["step_level"]["last_step_label_ratio"] = {
        k: v / len(last_step_labels) for k, v in last_step_counter.items()
    }
    
    # instance level
    all_instance_labels_list = [
        get_instance_label(instance) for instance in dataset
    ]
    
    stat["instance_level"]["num_instances"] = len(all_instance_labels_list)
    instance_level_counter = Counter(all_instance_labels_list)
    stat["instance_level"]["label_count"] = dict(instance_level_counter)
    stat["instance_level"]["label_ratio"] = {
        k: v / len(all_instance_labels_list)
        for k, v in instance_level_counter.items()
    }
    
    return stat
