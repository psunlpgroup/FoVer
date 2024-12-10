from tap import Tap
import json
import random

from src.path import get_fover_dataset_path
from src.config import splits_list
from src.dataset_creation.postprocessing.generate_final_dataset \
    import sample_dataset, get_final_dataset_name, sampled_dataset_size_list, \
    data_types_list
from src.dataset_creation.utils import get_prm_dataset_stats
from src.llm.utils import save_md5_hash


class MergeDatasetsTap(Tap):
    model_name: str
    dataset_names_list: list[str]


def main():
    args = MergeDatasetsTap().parse_args()

    for data_type in data_types_list:
        if data_type == "multi_turn_balanced_last_step":
            selected_splits = ["train"]
        else:
            selected_splits = splits_list

        for split in selected_splits:
            for size_name, _ in sampled_dataset_size_list:
                datasets_list = []
                for dataset_name in args.dataset_names_list:
                    # save final dataset
                    final_dataset_name = get_final_dataset_name(
                        dataset_name=dataset_name,
                        data_type=data_type,
                    ) + f"_{size_name}"

                    # load dataset
                    dataset_path = get_fover_dataset_path(
                        dataset_name=final_dataset_name,
                        model_name=args.model_name,
                        split=split,
                    )
                    with open(dataset_path, "r") as f:
                        dataset = [json.loads(line) for line in f.readlines()]
                    
                    datasets_list.append(dataset)
                
                # merge datasets
                merged_dataset = []
                for dataset in datasets_list:

                    if data_type == "multi_turn_balanced_last_step":
                        # last step ratio
                        last_step_correct_instances = [
                            d["error_labels"][-1] for d in dataset
                        ]
                        last_step_correct_ratio = \
                            sum(last_step_correct_instances) / len(dataset)
                        
                        # sample the dataset to the desired ratio
                        sampled_dataset = sample_dataset(
                            dataset=dataset,
                            target_data_size=int(
                                len(dataset) / len(datasets_list)),
                            last_step_correct_ratio=last_step_correct_ratio,
                        )
                    
                    else:
                        # instance correct ratio
                        correct_instances = [
                            d for d in dataset if all(d["error_labels"])]
                        instance_correct_ratio = \
                            len(correct_instances) / len(dataset)

                        sampled_dataset = sample_dataset(
                            dataset=dataset,
                            target_data_size=int(
                                len(dataset) / len(datasets_list)),
                            instance_correct_ratio=instance_correct_ratio,
                        )

                    merged_dataset.extend(sampled_dataset)
                
                # shuffle
                merged_dataset = random.Random(68).sample(
                    merged_dataset, len(merged_dataset)
                )

                # save final dataset
                final_dataset_name = get_final_dataset_name(
                    dataset_name="-".join(args.dataset_names_list),
                    data_type=data_type,
                ) + f"_{size_name}"
                merged_dataset_path = get_fover_dataset_path(
                    dataset_name=final_dataset_name,
                    model_name=args.model_name,
                    split=split,
                )
                merged_dataset_path.parent.mkdir(parents=True, exist_ok=True)
                with open(merged_dataset_path, "w") as f:
                    for instance in merged_dataset:
                        f.write(json.dumps(instance) + "\n")
                save_md5_hash(merged_dataset_path)
                print(
                    f"Saved merged dataset to {merged_dataset_path} with " \
                    f"{len(merged_dataset)} instances"
                )

                stats = get_prm_dataset_stats(merged_dataset)
                stats_path = merged_dataset_path.with_suffix(".stats.json")
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=4)
                
                # create variants only for multi-turn balanced last step
                if data_type != "multi_turn_balanced_last_step":
                    continue
                
                # simply add the dataset for 20k data
                # to make it 40k
                if size_name == "20k":
                    merged_dataset_40k = []
                    for dataset in datasets_list:
                        merged_dataset_40k.extend(dataset)
                    merged_dataset_40k = random.Random(68).sample(
                        merged_dataset_40k, len(merged_dataset_40k)
                    )
                    
                    # save final dataset
                    final_dataset_name = get_final_dataset_name(
                        dataset_name="-".join(args.dataset_names_list),
                        data_type=data_type,
                    ) + "_40k"
                    merged_dataset_path = get_fover_dataset_path(
                        dataset_name=final_dataset_name,
                        model_name=args.model_name,
                        split=split,
                    )
                    merged_dataset_path.parent.mkdir(
                        parents=True, exist_ok=True)
                    with open(merged_dataset_path, "w") as f:
                        for instance in merged_dataset_40k:
                            f.write(json.dumps(instance) + "\n")
                    save_md5_hash(merged_dataset_path)
                    print(
                        f"Saved merged dataset to {merged_dataset_path} with " \
                        f"{len(merged_dataset_40k)} instances"
                    )

                    # save stats
                    stats = get_prm_dataset_stats(merged_dataset_40k)
                    stats_path = merged_dataset_path.with_suffix(".stats.json")
                    with open(stats_path, "w") as f:
                        json.dump(stats, f, indent=4)
                    print(
                        f"Saved stats to {stats_path} with " \
                        f"{len(merged_dataset_40k)} instances"
                    )
                
                # make dataset for ablation study in dataset size
                # we duplicate the dataset for 5k, 10k and 20k
                # to make it 40k
                if size_name in ["5k", "10k", "20k"]:
                    multiple = {
                        "5k": 8,
                        "10k": 4,
                        "20k": 2,
                    }[size_name]

                    merged_dataset_duplicated = []
                    for _ in range(multiple):
                        merged_dataset_duplicated.extend(merged_dataset)

                        merged_dataset_duplicated = random.Random(68).sample(
                            merged_dataset_duplicated,
                            len(merged_dataset_duplicated)
                        )

                        # save final dataset
                        final_dataset_name = get_final_dataset_name(
                            dataset_name="-".join(args.dataset_names_list),
                            data_type=data_type,
                        ) + f"_{size_name}_duplicated_40k"
                        merged_dataset_path = get_fover_dataset_path(
                            dataset_name=final_dataset_name,
                            model_name=args.model_name,
                            split=split,
                        )
                        merged_dataset_path.parent.mkdir(
                            parents=True, exist_ok=True)
                        with open(merged_dataset_path, "w") as f:
                            for instance in merged_dataset_duplicated:
                                f.write(json.dumps(instance) + "\n")
                        save_md5_hash(merged_dataset_path)
                        print(
                            f"Saved merged dataset to {merged_dataset_path} " \
                            f"with {len(merged_dataset_duplicated)} instances"
                        )

                        # save stats
                        stats = get_prm_dataset_stats(
                            merged_dataset_duplicated
                        )
                        stats_path = merged_dataset_path.with_suffix(
                            ".stats.json"
                        )
                        with open(stats_path, "w") as f:
                            json.dump(stats, f, indent=4)
                        print(
                            f"Saved stats to {stats_path} with " \
                            f"{len(merged_dataset_duplicated)} instances"
                        )


                # make dataset for ablation study in label distribution
                if size_name == "20k":
                    for last_step_correct_ratio in [0.25, 0.75]:
                        inbalanced_merged_dataset = []
                        for dataset in datasets_list:
                            # sample the dataset to the desired ratio
                            sampled_dataset = sample_dataset(
                                dataset=dataset,
                                target_data_size=int(
                                    len(dataset) / len(datasets_list)),
                                last_step_correct_ratio=last_step_correct_ratio,
                            )

                            inbalanced_merged_dataset.extend(sampled_dataset)
                        
                        # shuffle
                        inbalanced_merged_dataset = random.Random(68).sample(
                            inbalanced_merged_dataset,
                            len(inbalanced_merged_dataset)
                        )

                        # save final dataset
                        final_dataset_name = get_final_dataset_name(
                            dataset_name="-".join(args.dataset_names_list),
                            data_type=data_type,
                        ) + f"_{size_name}_correct={last_step_correct_ratio}"

                        merged_dataset_path = get_fover_dataset_path(
                            dataset_name=final_dataset_name,
                            model_name=args.model_name,
                            split=split,
                        )
                        merged_dataset_path.parent.mkdir(
                            parents=True, exist_ok=True)
                        with open(merged_dataset_path, "w") as f:
                            for instance in inbalanced_merged_dataset:
                                f.write(json.dumps(instance) + "\n")
                        save_md5_hash(merged_dataset_path)
                        print(
                            f"Saved merged dataset to {merged_dataset_path} " \
                            f"with {len(inbalanced_merged_dataset)} instances"
                        )

                        # save stats
                        stats = get_prm_dataset_stats(
                            inbalanced_merged_dataset
                        )
                        stats_path = merged_dataset_path.with_suffix(
                            ".stats.json"
                        )
                        with open(stats_path, "w") as f:
                            json.dump(stats, f, indent=4)
                        print(
                            f"Saved stats to {stats_path} with " \
                            f"{len(inbalanced_merged_dataset)} instances"
                        )


if __name__ == "__main__":
    main()
