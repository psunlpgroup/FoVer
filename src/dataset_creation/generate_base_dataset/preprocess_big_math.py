import json

from tap import Tap
import datasets

from src.config import splits_list
from src.path import get_base_dataset_path
from src.llm.utils import save_md5_hash


class PreprocessBigMathTap(Tap):
    dataset_name: str | None = None  # dataset name
    domain: str | None = None  # domain name


dataset_size_dict = {
    "validation": 500,
    "test": 500,
}


domain_name_dict = {
    "math_word_problems": ["Mathematics -> Applied Mathematics -> Math Word Problems"],
}


remove_list = ["\\$", "\\%", "rd"]

def preprocess_answer_of_big_math(answer: str) -> str | None:
    for rm in remove_list:
        answer = answer.replace(rm, "")
    
    # if not int or float, return None
    try:
        float(answer)
    except ValueError:
        return None
    
    return answer


def preprocess_big_math_data(dataset: list[dict]) -> list[dict]:
    
    list_of_dict = []
    for d in dataset:
        processed_answer = preprocess_answer_of_big_math(d["answer"])
        
        # we only use simple cases whose answer is int or float
        if processed_answer is None:
            continue
        
        # add to list
        list_of_dict.append(
            {
                "id": d["id"],
                "question": d["problem"],
                "y_true": processed_answer,
            }
        )
    
    return list_of_dict    


def main():
    args = PreprocessBigMathTap().parse_args()
    
    # big-math only have train split
    # we will split it into train, validation, and test
    raw_data_hf = datasets.load_dataset(
        "SynthLabsAI/Big-Math-RL-Verified", split="train"
    )
    
    # save all data with data id
    all_data_path = get_base_dataset_path("bigmath", split="train")
    if all_data_path.exists():
        with open(all_data_path, "r") as f:
            raw_data = [json.loads(line) for line in f]
    else:
        all_data_path.parent.mkdir(parents=True, exist_ok=True)
        raw_data = []
        with open(all_data_path, "w") as f:
            for idx, d in enumerate(raw_data_hf):
                source = d["source"]
                data_id = f"bigmath_{source}_{idx:06d}"
                d["id"] = data_id
                
                raw_data.append(d)
                f.write(json.dumps(d) + "\n")       
        save_md5_hash(all_data_path)
    
    # filter out examples that are not in the target domain
    # and select diffcult cases (solve rate < 0.5)
    filtered_dataset: list[dict] = []
    for d in raw_data:
        if args.dataset_name is not None:
            if d["source"] != args.dataset_name:
                continue
        
        if args.domain is not None:
            if len(d["domain"]) != 1:
                continue
            
            if d["domain"][0] not in domain_name_dict[args.domain]:
                continue
        
        # remove too easy cases
        if d["llama8b_solve_rate"] is None:
            continue
        if d["llama8b_solve_rate"] >= 0.7:
            continue
        
        filtered_dataset.append(d)
    
    # convert to list of dict
    # and remove examples whose answer is not int or float
    list_of_dict = preprocess_big_math_data(filtered_dataset)

    # split into train, validation, and test
    if len(list_of_dict) < sum(dataset_size_dict.values()):
        raise ValueError(f"Not enough examples in {args.dataset_name}: {len(list_of_dict)} < {sum(dataset_size_dict.values())}")
    
    # split the dataset
    train_data = list_of_dict[:-sum(dataset_size_dict.values())]
    validation_data = list_of_dict[
        -(dataset_size_dict["validation"] + dataset_size_dict["test"]):
        -dataset_size_dict["test"]
    ]
    test_data = list_of_dict[-dataset_size_dict["test"]:]
    
    # save the dataset
    dataset_save_name = "bigmath"
    if args.dataset_name is not None:
        dataset_save_name += f"_{args.dataset_name}"
    if args.domain is not None:
        dataset_save_name += f"_{args.domain}"
    
    for split, data in zip(
                splits_list, [train_data, validation_data, test_data]
            ):
        
        # save to jsonl
        save_path = get_base_dataset_path(dataset_save_name, split=split)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")
        save_md5_hash(save_path)


if __name__ == "__main__":
    main()
