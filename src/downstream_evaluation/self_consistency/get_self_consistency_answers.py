import json
from collections import Counter

from src.path import get_downstream_evaluation_initial_responses_path, \
    get_majority_vote_output_path
from src.downstream_evaluation.sample_and_rank.generate_initial_responses import SampleAndRankPromptGenerationTap
from src.downstream_evaluation.evaluation.utils.extract_final_answer import extract_final_answer_for_downstream_evaluation
from src.llm.utils import save_md5_hash


class SelfConsistencyTap(SampleAndRankPromptGenerationTap):
    sample_k: int  # number of responses to generate


def main():
    args = SelfConsistencyTap().parse_args()
    
    all_final_answers: list[list[list]] = []
    for sample_idx in range(args.sample_k):
        initial_responses_path = get_downstream_evaluation_initial_responses_path(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            split="test",
            prompt_type="few-shot",
            sample_idx=sample_idx,
        )
        with open(initial_responses_path.with_suffix(".postprocessed.jsonl"), "r") as f:
            initial_responses = [json.loads(line) for line in f]
        
        # initialize
        if sample_idx == 0:
            all_final_answers = [[None, []] for _ in initial_responses]
        
        # extract final answers
        for idx, d in enumerate(initial_responses):
            all_final_answers[idx][0] = d["id"]
            all_final_answers[idx][1].append(
                extract_final_answer_for_downstream_evaluation(
                    dataset_name=args.dataset_name, prediction=d
                )
            )
    
    # save majority vote
    majority_vote = []
    for idx, final_answers in enumerate(all_final_answers):
        majority = Counter(final_answers[1]).most_common()[0][0]
        majority_vote.append(
            {
                "id": final_answers[0],
                "y_pred": majority,
            }
        )
    
    # save majority vote
    majority_vote_path = get_majority_vote_output_path(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        split="test",
        prompt_type="few-shot",
    )
    majority_vote_path.parent.mkdir(parents=True, exist_ok=True)
    with open(majority_vote_path, "w") as f:
        for d in majority_vote:
            f.write(json.dumps(d) + "\n")
    save_md5_hash(majority_vote_path)


if __name__ == "__main__":
    main()
