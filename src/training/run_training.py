import os
import subprocess
import time
import shutil
from typing import Literal
from pathlib import Path

from tap import Tap
import torch
import transformers
from llama_recipes.model_checkpointing.checkpoint_handler import load_sharded_model_single_gpu

from src.typing import BASE_MODEL, OPTIMIZERS
from src.path import finetuned_models_dir, get_fover_dataset_path
from src.llm.llama_recipes_updated.utils.train_utils import get_save_folder


class TrainingTap(Tap):
    model_name: str # BASE_MODEL  # e.g., meta-llama/Llama-3.1-8B-Instruct
    dataset_name: str  # e.g. fldx2_symbol
    optimizer: OPTIMIZERS = "RecAdam"  # "RecAdam" or "AdamW"
    num_epochs: int = 1
    accumulated_batch_size: int = 256
    rec_adam_fisher_coef: int = 4000
    learning_rate: str = "2e-05"
    use_lora: bool = False


class PseudoArgsForGetSaveFolder(Tap):
    model_name: str
    dist_checkpoint_root_folder: str
    dist_checkpoint_folder: str


def main():
    args = TrainingTap().parse_args()
    print(args)
    
    # get CUDA_VISIBLE_DEVICES
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        first_device = 0
    else:
        first_device = int(visible_devices.split(",")[0])
    master_port = 34229 + first_device
    print(f"MASTER_PORT: {master_port}")
    
    command = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "--master_port", f"{master_port}",
        "src/llm/run_training.py"
    ]
    
    command.extend(["--enable_fsdp"])
    command.extend(["--use_fast_kernels"])
    command.extend(["--fsdp_config.pure_bf16"])
    
    # recall adam optimizer
    command.extend(["--optimizer", args.optimizer])
    command.extend(["--rec_adam_fisher_coef", str(args.rec_adam_fisher_coef)])
    
    # command.extend(["--model_name_or_path", f"{args.model_name}"])
    command.extend(["--model_name", args.model_name])
    command.extend(["--num_epochs", str(args.num_epochs)])

    # # training data will be loaded from local files
    # data_dir = get_preprocessed_training_data_path(args.dataset_name, args.model_name, "train").parent
    # command.extend(["--dataset_name", f"{data_dir}"])
    
    # custom data path
    if args.model_name == "meta-llama/Llama-3.1-8B-Instruct":
        custom_dataset_path = f"src/training/custom_dataset_files/custom_dataset_llama31_8B_{args.dataset_name}.py"
    elif args.model_name == "Qwen/Qwen2-7B-Instruct":
        custom_dataset_path = f"src/training/custom_dataset_files/custom_dataset_qwen2_7B_{args.dataset_name}.py"
    elif args.model_name == "Qwen/Qwen2.5-7B-Instruct":
        custom_dataset_path = f"src/training/custom_dataset_files/custom_dataset_qwen25_7B_{args.dataset_name}.py"
    elif args.model_name == "google/gemma-2-9b-it":
        custom_dataset_path = f"src/training/custom_dataset_files/custom_dataset_gemma2_9B_{args.dataset_name}.py"
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    # https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning/datasets#training-on-custom-data
    command.extend(["--dataset", "custom_dataset"])
    command.extend(["--custom_dataset.file", custom_dataset_path])
    
    # timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    short_dataset_name = args.dataset_name.split("/")[-1]
    short_model_name = args.model_name.split("/")[-1]
    # output_dir = finetuned_models_dir / short_dataset_name / f"{short_model_name}-{timestamp}"
    dist_checkpoint_root_folder = f"{finetuned_models_dir / args.optimizer / short_dataset_name}"
    command.extend(["--dist_checkpoint_root_folder", dist_checkpoint_root_folder])
    dist_checkpoint_folder = f"{short_model_name}-{timestamp}"
    command.extend(["--dist_checkpoint_folder", dist_checkpoint_folder])
    
    command.extend(["--output_dir", f"{dist_checkpoint_root_folder}/{dist_checkpoint_folder}"])
    command.edtend(["--profiler_dir", f"{dist_checkpoint_root_folder}/{dist_checkpoint_folder}"])

    # batch size
    # gradient_accumulation_steps should be set so that the effective batch size is args.accumulated_batch_size
    num_devices = torch.cuda.device_count()
    print(f"num_devices: {num_devices}")
    
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if "A100" in gpu_name:
        per_device_batch_size = 4
    elif "A6000" in gpu_name:
        per_device_batch_size = 2
    elif "A5000" in gpu_name:
        per_device_batch_size = 1
    else:
        raise ValueError(f"Unsupported GPU: {gpu_name}")
    
    if args.model_name == "google/gemma-2-9b-it":
        per_device_batch_size //= 2
    
    print(f"GPU: {gpu_name}")
    print(f"per_device_batch_size: {per_device_batch_size}")
    
    if args.accumulated_batch_size % (per_device_batch_size * num_devices) != 0:
        raise ValueError(f"accumulated_batch_size ({args.accumulated_batch_size}) must " \
                          "be divisible by the product of per_device_batch_size " \
                          "({per_device_batch_size}) and the number of devices ({num_devices})")
    gradient_accumulation_steps = args.accumulated_batch_size // (per_device_batch_size * num_devices)
    command.extend(["--batch_size_training", f"{per_device_batch_size}"])
    # command.extend(["--per_device_train_batch_size", f"{per_device_batch_size}"])
    # command.extend(["--per_device_eval_batch_size", f"{per_device_batch_size}"])
    command.extend(["--gradient_accumulation_steps", f"{gradient_accumulation_steps}"])
    
    # lora
    if args.use_lora:
        command.append("--use_peft")
        command.extend(["--peft_method", "lora"])
    
    # dataset config
    command.extend(["--trust_remote_code"])
    
    # training config
    command.extend(["--lr", args.learning_rate])
    
    # command.extend(["--learning_rate", "2e-05"])
    
    # 50% warmup is directly implemented in the training code
    # command.extend(["--warmup_steps", "200"])

    # I am working on this    
    # command.extend(["--run_validation_steps", "200"])
    
    command.extend(["--save_metrics", "True"])
    
    # run command
    subprocess.run(command)

    # model is already saved by the train function.
    # However, if StateDictType.SHARDED_STATE_DICT, the pretrained model is saved in a sharded state dict format
    # and we need to load the model using the load_model_from_sharded_state_dict function.
    # Therefore, we convert the saved model to a usual format
    print("Converting the sharded state dict model to the 'save_pretrained' format")
    if args.model_name == "meta-llama/Llama-3.1-8B-Instruct":
        model = transformers.LlamaForCausalLM.from_pretrained(args.model_name)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    pseudo_args = PseudoArgsForGetSaveFolder().parse_args(
        [
            "--model_name", args.model_name,
            "--dist_checkpoint_root_folder", dist_checkpoint_root_folder,
            "--dist_checkpoint_folder", dist_checkpoint_folder
        ]
    )
    model_saved_folder = get_save_folder(pseudo_args)
    pretrained_model = load_sharded_model_single_gpu(model, model_saved_folder)
    pretrained_model.save_pretrained(model_saved_folder)
    
    # remove the sharded state dict model (all files of model_saved_folder/*.distcp)
    for file in os.listdir(model_saved_folder):
        if file.endswith(".distcp"):
            os.remove(os.path.join(model_saved_folder, file))
    
    # copy train data hash
    train_data_hash_path = get_fover_dataset_path(
        dataset_name=args.dataset_name, model_name=args.model_name,
        split="train"
    ).with_suffix(".jsonl.md5")
    new_train_data_hash_path = \
        Path(model_saved_folder) / train_data_hash_path.name
    shutil.copy(train_data_hash_path, new_train_data_hash_path)

    # move save folder
    # the original save folder includes the full model name 
    # (e.g., meta-llama/Llama-3.1-8B-Instruct) at the end of the folder name
    # we remove the model name and add the optimizer name
    new_save_folder = f"{model_saved_folder[:-(len(args.model_name)+1)]}_{args.dataset_name}_{args.optimizer}"
    if args.optimizer == "RecAdam":
        new_save_folder += f"_{args.rec_adam_fisher_coef}"
    new_save_folder += f"_{args.learning_rate}"
    
    print(f"Moving the model from {model_saved_folder} to {new_save_folder}")
    shutil.move(model_saved_folder, new_save_folder)
    
    if "/" in args.model_name:
        # unnecessary directory
        model_short_name = args.model_name.split("/")[1]
        unnecessary_dir = model_saved_folder[:-(len(model_short_name)+1)]
        print(f"Removing the unnecessary directory {unnecessary_dir}")
        os.rmdir(unnecessary_dir)


if __name__ == "__main__":
    main()
