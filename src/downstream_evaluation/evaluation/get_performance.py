import json

from tap import Tap

from src.path import get_downstream_evaluation_metrics_path
from src.downstream_evaluation.evaluation.utils import \
    get_performance_for_downstream_evaluation


class DownstreamPerformanceTap(Tap):
    dataset_name: str  # dataset name
    base_model_name: str  # base model name
    prediction_path: str  # path to the JSONL file with the model predictions


def main():
    args = DownstreamPerformanceTap().parse_args()
    
    with open(args.prediction_path, "r") as f:
        predictions = [json.loads(line) for line in f]
    
    performance = get_performance_for_downstream_evaluation(
        dataset_name=args.dataset_name,
        predictions=predictions
    )
    
    # without filtering
    # save performance
    evaluation_metrics_path = get_downstream_evaluation_metrics_path(
        dataset_name=args.dataset_name,
        model_name=args.base_model_name,
        prediction_path=args.prediction_path,
        split="test"
    )
    
    evaluation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evaluation_metrics_path, "w") as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    main()
