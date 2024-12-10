import json

from src.path import get_fover_dataset_path, tables_dir
from src.config import splits_list, base_model_names


def main():
    # latex table for error perceptage
    table = []
    for base_model_name in base_model_names:
        for dataset_name in ["fldx2_symbol_multi_turn_10k", "isabelle_all_multi_turn_10k"]:
            row = [
                f"{base_model_name:35s}",
                f"{dataset_name:30s}",
            ]

            for split in splits_list:
                stat_path = get_fover_dataset_path(
                    dataset_name=dataset_name,
                    model_name=base_model_name,
                    split=split,
                ).with_suffix(".stats.json")
                with open(stat_path, "r") as f:
                    stats = json.load(f)

                for level_name in ["instance", "step"]:
                    num_instance = stats[f"{level_name}_level"][
                        f"num_{level_name}s"]
                    row.append(f"{num_instance:5d}")

                    error_percentage = stats[f"{level_name}_level"][
                        "label_ratio"]["false"] * 100
                    row.append(f"{error_percentage:5.1f}\\%")
            
            table.append(row)
        
    table_path = tables_dir / "dataset_statistics" / "error_percentage.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    with open(table_path, "w") as f:
        for row_list in table:
            row_str = " & ".join(row_list)
            f.write(f"{row_str} \\\\\n")


if __name__ == "__main__":
    main()
