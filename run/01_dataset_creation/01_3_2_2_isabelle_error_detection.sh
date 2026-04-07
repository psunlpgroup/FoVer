for MODEL_NAME in "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct"
do

export MODEL=$MODEL_NAME
export CONVERSION_MODEL=$MODEL_NAME
export FORMAL_PROOF_MODEL=$MODEL_NAME

conda activate fover_data_creation

for BASE_DATASET in gsm8k bigmath_math_word_problems metamathqa_gsm8k
do

# run this script for each dataset, model, and split

for SPLIT in train validation test
do

# for training, there are 64 batches
# for test and validation, there are 16 batches
# depending on cpus, you can run multiple batches in parallel
# here is an example of running 16 batches in parallel
export BATCH_START=0
export BATCH_END=15

# automatic verification using isabelle
python src/dataset_creation/base_dataset_specific/isabelle/error_detection/run_error_detection_in_parallel.py \
    --base_model_name $MODEL --dataset_name $BASE_DATASET \
    --conversion_model_name $CONVERSION_MODEL --formal_proof_generation_model_name $FORMAL_PROOF_MODEL \
    --split $SPLIT \
    --batch_idx_start $BATCH_START --batch_idx_end $BATCH_END \
    # --overwrite_results

done
done
done
