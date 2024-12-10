export CONVERSION_MODEL=meta-llama/Llama-3.3-70B-Instruct

conda activate fover_data_creation

# run this script for each dataset, model, and split

export BASE_DATASET=metamathqa_gsm8k # gsm8k bigmath_math_word_problems metamathqa_gsm8k
export MODEL=meta-llama/Llama-3.1-8B-Instruct # meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
export SPLIT=validation # train validation test

# for training, there are 64 batches
# for test and validation, there are 16 batches
# depending on cpus, you can run multiple batches in parallel
# here is an example of running 16 batches in parallel
export BATCH_START=0
export BATCH_END=15

# automatic verification using isabelle
python src/dataset_creation/base_dataset_specific/isabelle/error_detection/run_error_detection_in_parallel.py \
    --base_model_name $MODEL --dataset_name $BASE_DATASET \
    --conversion_model_name $CONVERSION_MODEL --split $SPLIT \
    --batch_idx_start $BATCH_START --batch_idx_end $BATCH_END \
    # --overwrite_results
