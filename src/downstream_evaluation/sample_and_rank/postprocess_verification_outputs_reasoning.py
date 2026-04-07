"""Postprocess reasoning-style verification outputs for sample-and-rank evaluation."""

import json
import re
from typing import List, Optional

from src.path import (
    get_downstream_evaluation_initial_responses_path,
    get_verification_for_sample_and_rank_outputs_path,
    get_verification_scores_for_sample_and_rank_path,
)
from src.downstream_evaluation.sample_and_rank.run_verification import (
    EvaluationForSampleAndRankTap,
)
from src.downstream_evaluation.utils import get_solution_steps_from_response
from src.llm.utils import save_md5_hash
from src.prm.preprocessing import ANSWER_START, ANSWER_END
from src.utils.prm.postprocess import get_postprocessed_prm_output_format


def is_reasoning_prm_final_answer_correct(response: str) -> Optional[bool]:
    """Return True if the final <ANSWER> tag marks the step correct, False if incorrect."""
    if not response:
        return None

    pattern = re.compile(
        re.escape(ANSWER_START) + r"\s*(correct|incorrect)\s*" + re.escape(ANSWER_END),
        re.IGNORECASE,
    )
    matches = pattern.findall(response)
    if not matches:
        return None

    verdict = matches[-1].lower()
    if verdict == "correct":
        return True
    if verdict == "incorrect":
        return False
    return None


def main():
    args = EvaluationForSampleAndRankTap().parse_args()

    if args.verification_prompt_type != "reasoning":
        raise ValueError(
            "postprocess_verification_outputs_reasoning.py expects --verification_prompt_type reasoning"
        )

    for verification_score_type in ["num_incorrect_steps"]:
        verification_scores_path = get_verification_scores_for_sample_and_rank_path(
            dataset_name=args.dataset_name,
            base_model_name=args.initial_generation_model_name,
            verification_model_name=args.verification_model_name,
            verification_score_type=verification_score_type,
            split="test",
            prompt_type="reasoning",
            few_shot_verification=args.few_shot_verification,
        )
        verification_scores_path.parent.mkdir(parents=True, exist_ok=True)

        for sample_idx in range(args.sample_k):
            initial_responses_path = get_downstream_evaluation_initial_responses_path(
                dataset_name=args.dataset_name,
                model_name=args.initial_generation_model_name,
                split="test",
                prompt_type="few-shot",
                sample_idx=sample_idx,
            )
            with open(initial_responses_path, "r") as f:
                initial_responses = [json.loads(line) for line in f]

            verification_outputs_path = get_verification_for_sample_and_rank_outputs_path(
                dataset_name=args.dataset_name,
                initial_response_model_name=args.initial_generation_model_name,
                verification_model_name=args.verification_model_name,
                split="test",
                prompt_type="reasoning",
                sample_idx=sample_idx,
                few_shot_verification=args.few_shot_verification,
            )
            with open(verification_outputs_path, "r") as f:
                verification_outputs = [json.loads(line) for line in f]

            processed_results = []
            output_offset = 0

            for response_entry in initial_responses:
                solution_steps = get_solution_steps_from_response(response_entry["response"])
                num_steps = len(solution_steps)

                if output_offset + num_steps > len(verification_outputs):
                    raise ValueError(
                        "Mismatch between verification outputs and expected step counts."
                    )

                step_outputs = verification_outputs[output_offset : output_offset + num_steps]
                output_offset += num_steps

                step_predictions: List[Optional[bool]] = []
                step_ids: List[str] = []
                for step_output in step_outputs:
                    step_ids.append(step_output["id"])
                    step_predictions.append(
                        is_reasoning_prm_final_answer_correct(step_output["response"])
                    )

                pseudo_y_true = [None] * num_steps
                postprocessed = get_postprocessed_prm_output_format(
                    y_pred_step_level=step_predictions,
                    y_true_step_level=pseudo_y_true,
                    verification_score_type=verification_score_type,
                )

                processed_results.append(postprocessed)

            if output_offset != len(verification_outputs):
                raise ValueError(
                    "Verification outputs contain extra entries beyond the expected step counts."
                )

            idx_scores_path = verification_scores_path.with_suffix(
                f".intermediate.idx={sample_idx}.jsonl"
            )
            with open(idx_scores_path, "w") as f:
                for result in processed_results:
                    f.write(json.dumps(result) + "\n")
            save_md5_hash(idx_scores_path)


if __name__ == "__main__":
    main()
