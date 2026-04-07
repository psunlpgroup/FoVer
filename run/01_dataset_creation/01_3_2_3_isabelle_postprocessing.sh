conda activate fover_data_creation

# collect results from other servers to intermediate_outputs/isabelle

# postprocessing
for BASE_DATASET in gsm8k bigmath_math_word_problems metamathqa_gsm8k
do
    for MODEL in Qwen/Qwen2.5-7B-Instruct
    do
        export CONVERSION_MODEL=$MODEL
        python src/dataset_creation/base_dataset_specific/isabelle/error_detection/postprocessing.py \
            --base_model_name $MODEL --dataset_name $BASE_DATASET \
            --conversion_model_name $CONVERSION_MODEL \
            --formal_proof_generation_model_name $MODEL
    done
done

# merge datasets
python src/dataset_creation/base_dataset_specific/isabelle/error_detection/merge_error_labels_from_multiple_datasets.py
