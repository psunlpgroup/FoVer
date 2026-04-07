import json
import random
from pathlib import Path
from typing import Any, Dict, List

from tap import Tap

from src.dataset_creation.utils.get_dataset_stats import get_prm_dataset_stats
from src.llm.utils import save_md5_hash


class DownsampleSimpleArgs(Tap):
    """Command-line arguments for simple dataset downsampling and upsampling."""

    input_path: Path  # JSONL dataset file to read
    output_path_10k: Path  # Destination for the 10k-derived (upsampled) dataset
    output_path_20k: Path  # Destination for the 20k-derived (upsampled) dataset
    shuffle: bool = True  # Shuffle records before writing outputs
    seed: int = 68  # RNG seed for reproducibility


def main() -> None:
    # Parse CLI flags for dataset paths and RNG settings.
    args = DownsampleSimpleArgs().parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_path}")

    # Load the JSONL dataset into memory and validate contents.
    dataset: List[Dict[str, Any]] = []
    with args.input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as exc:
                message = (
                    f"Failed to parse {args.input_path} (line {line_number}): {exc}"
                )
                raise ValueError(message) from exc
    if not dataset:
        raise ValueError("Input dataset is empty.")

    original_size = len(dataset)
    # Build the fixed downsample target configurations.
    targets = [(10_000, args.output_path_10k), (20_000, args.output_path_20k)]

    processed_outputs: List[str] = []

    for target_size, output_path in targets:
        if target_size <= 0:
            raise ValueError(f"Downsample size must be positive: {target_size}")
        if target_size > original_size:
            raise ValueError(
                f"Downsample size {target_size} exceeds dataset size {original_size}."
            )

        # Sample a deterministic subset and replicate it up to the original size.
        sample_seed = args.seed + target_size
        sample_rng = random.Random(sample_seed)
        downsampled = sample_rng.sample(dataset, target_size)

        upsampled: List[Dict[str, Any]] = []
        while len(upsampled) + len(downsampled) <= original_size:
            upsampled.extend(downsampled)
        if len(upsampled) < original_size:
            additional_size = original_size - len(upsampled)
            upsampled.extend(random.Random(additional_size).sample(downsampled, additional_size))
        assert len(upsampled) == original_size

        # Optionally reshuffle the upsampled records before writing.
        if args.shuffle:
            shuffle_seed = args.seed + target_size + original_size
            random.Random(shuffle_seed).shuffle(upsampled)

        # Persist the upsampled dataset and corresponding checksum.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in upsampled:
                handle.write(json.dumps(record) + "\n")
        save_md5_hash(output_path)

        # Store aggregated statistics alongside metadata describing this run.
        stats = get_prm_dataset_stats(upsampled)
        stats.update(
            {
                "subset_type": "upsample",
                "base_subset_size": target_size,
                "output_size": original_size,
                "source_size": original_size,
                "source_path": str(args.input_path),
                "sample_seed": sample_seed,
            }
        )
        stats_path = output_path.with_suffix(".stats.json")
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        processed_outputs.append(str(output_path))

    # Emit a concise summary of generated outputs for logging.
    print(
        "Generated upsampled datasets from downsample sizes 10k and 20k: "
        + ", ".join(processed_outputs)
    )


if __name__ == "__main__":
    main()
