from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SuccessStats:
    total: int
    success_true: int

    @property
    def success_false(self) -> int:
        return self.total - self.success_true

    @property
    def proportion_true(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success_true / self.total

    @property
    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success_false / self.total


def find_result_files(base_dir: Path) -> Iterable[Path]:
    """Yield every all_sorry.result.json file under base_dir."""
    pattern = "all_sorry.result.json"
    return base_dir.rglob(pattern)


def compute_stats(files: Iterable[Path]) -> SuccessStats:
    total = 0
    success_true = 0

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        success_value = payload.get("success")
        if isinstance(success_value, bool):
            total += 1
            if success_value:
                success_true += 1

    return SuccessStats(total=total, success_true=success_true)


def write_stats(stats: SuccessStats, output_path: Path, source_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "source_directory": str(source_dir),
        "total_instances": stats.total,
        "success_true": stats.success_true,
        "success_false": stats.success_false,
        "success_proportion_true": stats.proportion_true,
        "error_rate": stats.error_rate,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)
    print(f"Wrote stats to {output_path}")


def main() -> None:
    source_dir = Path(
        "intermediate_outputs/isabelle/formal_proofs/gsm8k/"
        "initial_generation=Qwen2.5-7B-Instruct/conversion=Qwen2.5-7B-Instruct/"
        "formal_proof_generation=Qwen2.5-7B-Instruct/train"
    )

    output_path = Path("analysis/dataset_statistics/formal_solution_stats.json")

    stats = compute_stats(find_result_files(source_dir))
    write_stats(stats=stats, output_path=output_path, source_dir=source_dir)


if __name__ == "__main__":
    main()
