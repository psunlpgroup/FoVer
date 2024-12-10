from pathlib import Path

from src.config import base_model_names, \
    train_dataset_names_list, train_dataset_names_list_multi_turn, \
    get_fover_dataset_name


batch_size_for_dataset = {
    "fldx2_symbol_with_cot": 256,
    "isabelle_all_with_cot": 64,  # dataset size is small
    "fldx2_symbol_with_cot,isabelle_all_with_cot": 256,
    #
    "fldx2_symbol_multi_turn_10k": 32,
    "isabelle_all_multi_turn_10k": 32,
    "fldx2_symbol-isabelle_all_multi_turn_10k": 32,
    #
    "fldx2_symbol_multi_turn_balanced_last_step_20k": 32,
    "isabelle_all_multi_turn_balanced_last_step_20k": 32,
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k": 32,
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k": 32,
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25": 32,
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75": 32,
    #
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k": 32,
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k": 32,
    "fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k": 32,
    #
    "fldx2_symbol_multi_turn": 256,
    "isabelle_all_multi_turn": 64,  # dataset size is small
    "fldx2_symbol_multi_turn,isabelle_all_multi_turn": 256,
}


base_config_file_path = Path("src/llama_factory/base_config_file.yaml")
llama_factory_config_dir = Path("llama_factory_config")

def get_yaml_file_path(model_name: str, dataset_name: str,
                       learning_rate: str) -> Path:
    return llama_factory_config_dir / \
        f"{model_name}_{dataset_name}_{learning_rate}.yaml"


def main():
    llama_factory_config_dir.mkdir(exist_ok=True)
    
    num_gpus = 4
    
    with open(base_config_file_path, "r") as file:
        base_config = file.read()
    
    for model_name in base_model_names:
        for train_dataset_name in train_dataset_names_list \
                + train_dataset_names_list_multi_turn:
            # batch size
            if model_name == "google/gemma-2-9b-it":
                per_device_batch_size = 2
            else:
                per_device_batch_size = 4
            
            gradient_accumulation_steps = \
                batch_size_for_dataset[train_dataset_name] // (per_device_batch_size * num_gpus)
            
            model_short_name = model_name.split("/")[-1]
            
            hf_dataset_names_list: list[str] = []
            for sub_dataset_name in train_dataset_name.split(","):
                hf_dataset_name = get_fover_dataset_name(
                    base_model_name=model_name,
                    base_dataset_name=sub_dataset_name
                ).split("/")[-1]
                
                hf_dataset_names_list.append(hf_dataset_name)
            
            # validation dataset name
            if "correct=" in ",".join(hf_dataset_names_list) or "duplicated" in ",".join(hf_dataset_names_list):
                short_model_name = model_name.split("/")[-1]
                validation_dataset_name = f"FoVer_PRM_FormalLogic-FormalProof_10k_{short_model_name}_validation"
            else:
                validation_dataset_name = ",".join(
                        [name.replace("_balanced_last_step", "").replace("20k", "10k").replace("40k", "10k")  + "_validation"
                         for name in hf_dataset_names_list]
                    )
            
            for learning_rate in ["1.0e-6", "2.0e-6", "5.0e-6", "1.0e-5", "2.0e-5"]:
                output_dir = f"../FoVer/llama_factory_finetuned_models/{model_short_name}_{train_dataset_name}_{learning_rate}"
                
                # fill in the config template
                config = base_config.format(
                    model_name=model_name,
                    dataset_name=",".join(hf_dataset_names_list),
                    output_dir=output_dir,
                    per_device_train_batch_size=per_device_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    validation_dataset_name=validation_dataset_name,
                    learning_rate=learning_rate,
                    mask_history=True \
                        if "balanced_last_step" in train_dataset_name \
                        else False,
                )
                
                # save config to yaml file
                yaml_file_path = get_yaml_file_path(
                    model_short_name, train_dataset_name,
                    learning_rate=learning_rate
                )
                with open(yaml_file_path, "w") as file:
                    file.write(config)


if __name__ == '__main__':
    main()
