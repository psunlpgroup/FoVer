from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from tap import Tap

from src.dataset_creation.utils.get_dataset_stats import get_prm_dataset_stats
from src.llm.utils import save_md5_hash


class MergeDatasetsSimpleArgs(Tap):
    """Command-line arguments for simple dataset merging."""

    input_paths: List[Path]  # JSONL dataset files to merge
    output_path: Path  # Destination JSONL file
    shuffle: bool = True  # Shuffle merged instances before writing
    seed: int = 68  # RNG seed when shuffling


def main() -> None:
    args = MergeDatasetsSimpleArgs().parse_args()

    if not args.input_paths:
        raise ValueError("At least one --input-paths value is required.")

    merged_dataset: List[Dict[str, Any]] = []  # Collect merged instances
    source_counts: Dict[str, int] = {}  # Record how many entries came from each source

    # Load each source dataset and accumulate entries
    for input_path in args.input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {input_path}")
        dataset = []  # Buffer for the current source file
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    message = f"Failed to parse {input_path} (line {line_number}): {exc}"
                    raise ValueError(message) from exc
        merged_dataset.extend(dataset)
        source_counts[str(input_path)] = len(dataset)

    if not merged_dataset:
        raise ValueError("No data found across provided input datasets.")

    # Optionally shuffle the merged dataset deterministically
    if args.shuffle:
        random.Random(args.seed).shuffle(merged_dataset)

    # Write merged dataset and record its checksum
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in merged_dataset:
            handle.write(json.dumps(record) + "\n")
    save_md5_hash(args.output_path)

    # Compute and store dataset statistics alongside per-source counts
    stats = get_prm_dataset_stats(merged_dataset)
    stats["sources"] = source_counts
    stats_path = args.output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(
        f"Merged {len(args.input_paths)} datasets into {args.output_path} "
        f"({len(merged_dataset)} instances)."
    )


if __name__ == "__main__":
    main()
