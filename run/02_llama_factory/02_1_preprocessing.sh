# python src/llama_factory/convert_datasets.py

cp ../LLaMA-Factory-FoVer/data/dataset_info.json ../LLaMA-Factory-FoVer-Qwen3/data/dataset_info.json


# fover
python src/llama_factory/convert_datasets_simple.py \
    --input_jsonl fover_dataset/fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k/merged-Llama-3.1-8B-Instruct-Qwen2.5-7B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl \
    --dataset_info_key FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_202512

python src/llama_factory/generate_yaml_files.py \
    --train_dataset_name FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_202512 \
    --num_gpus 4
