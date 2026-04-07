from pathlib import Path

from tap import Tap

from src.config import base_model_names, get_fover_dataset_info_key


batch_size = 32

base_config_file_path = Path("src/llama_factory/base_config_file.yaml")
llama_factory_config_dir = Path("llama_factory_config")


class GenerateYamlFilesArgs(Tap):
    """CLI arguments for generating LLaMA Factory YAML configs."""

    train_dataset_name: str  # Target FoVer training dataset name
    num_gpus: int = 4  # Number of GPUs considered when computing batch size


def get_yaml_file_path(model_name: str, dataset_name: str,
                       learning_rate: str) -> Path:
    
    name = f"{model_name}_{dataset_name}"
    name += f"_{learning_rate}"
    
    return llama_factory_config_dir / f"{name}.yaml"


def main() -> None:
    args = GenerateYamlFilesArgs().parse_args()

    llama_factory_config_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = args.num_gpus

    with open(base_config_file_path, "r", encoding="utf-8") as file:
        base_config = file.read()

    for model_name in base_model_names:
        # batch size
        per_device_batch_size = 2 if model_name == "google/gemma-2-9b-it" else 4

        gradient_accumulation_steps = (
            batch_size
            // (per_device_batch_size * num_gpus)
        )

        model_short_name = model_name.split("/")[-1]
        mask_history = "True" if "balanced_last_step" in args.train_dataset_name else "False"

        for learning_rate in ["1.0e-6", "2.0e-6", "5.0e-6", "1.0e-5", "2.0e-5"]:
            output_dir = (
                f"../FoVer/llama_factory_finetuned_models/"
                f"{model_short_name}_{args.train_dataset_name}_{learning_rate}"
            )

            # fill in the config template
            config = base_config.format(
                model_name=model_name,
                dataset_name=args.train_dataset_name,
                output_dir=output_dir,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                mask_history=mask_history,
            )

            # save config to yaml file
            yaml_file_path = get_yaml_file_path(
                model_short_name,
                args.train_dataset_name,
                learning_rate=learning_rate,
            )
            with open(yaml_file_path, "w", encoding="utf-8") as file:
                file.write(config)


if __name__ == '__main__':
    main()
