export CONVERSION_MODEL=meta-llama/Llama-3.3-70B-Instruct

conda activate fover_data_creation

# collect results from other servers to intermediate_outputs/isabelle

# postprocessing
for BASE_DATASET in gsm8k bigmath_math_word_problems metamathqa_gsm8k
do
    for MODEL in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
    do
        python src/dataset_creation/base_dataset_specific/isabelle/error_detection/postprocessing.py \
            --base_model_name $MODEL --dataset_name $BASE_DATASET \
            --conversion_model_name $CONVERSION_MODEL
    done
done

# merge datasets
python src/dataset_creation/base_dataset_specific/isabelle/error_detection/merge_error_labels_from_multiple_datasets.py
