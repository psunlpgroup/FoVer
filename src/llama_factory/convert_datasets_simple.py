from __future__ import annotations

import json
from pathlib import Path

import datasets
from tap import Tap

from src.llama_factory.convert_datasets import (
    convert_to_sharedgpt,
    llama_factory_datasets_dir,
    dataset_json_path,
)
from src.llm.utils import save_md5_hash, save_time_stamp


class ConvertDatasetsSimpleArgs(Tap):
    """Arguments for converting a single JSONL dataset to ShareGPT format."""

    input_jsonl: Path  # Path to the source JSONL dataset
    dataset_info_key: str  # Key used to identify the dataset in dataset_info.json
    split: str = "train"  # Dataset split name used for folder placement


def main() -> None:
    args = ConvertDatasetsSimpleArgs().parse_args()

    # Ensure the source dataset exists before proceeding
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_jsonl}")

    # Load the source JSONL file and convert it to ShareGPT format
    dataset = datasets.load_dataset("json", data_files=str(args.input_jsonl))["train"]
    converted_dataset = convert_to_sharedgpt(dataset)

    # Prepare the output location for the converted split
    split_dir = llama_factory_datasets_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Persist the converted dataset and its checksum
    dataset_path = split_dir / f"{args.dataset_info_key}.json"
    with dataset_path.open("w", encoding="utf-8") as handle:
        json.dump(converted_dataset, handle, indent=4)
    save_md5_hash(dataset_path)
    save_time_stamp(dataset_path)

    # Store simple statistics for the converted dataset
    stats = {"num_samples": len(converted_dataset)}
    stats_path = dataset_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    # Update or create dataset_info.json with the new entry
    dataset_info_path = dataset_json_path
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_info: dict[str, dict]
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as handle:
            dataset_info = json.load(handle)
    else:
        dataset_info = {}

    dataset_info[args.dataset_info_key] = {
        "file_name": str(Path("../../FoVer") / dataset_path),
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    }

    # Persist updated dataset metadata
    with dataset_info_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2)

    print(
        f"Converted {args.input_jsonl} to {dataset_path} with {len(converted_dataset)} samples."
    )


if __name__ == "__main__":
    main()
