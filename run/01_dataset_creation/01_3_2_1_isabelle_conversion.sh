# code for formal theorem proving dataset creation

export CONVERSION_MODEL=meta-llama/Llama-3.3-70B-Instruct

conda activate llm-inference

for BASE_DATASET in gsm8k bigmath_math_word_problems metamathqa_gsm8k
do
    for MODEL in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
    do
        python src/dataset_creation/base_dataset_specific/isabelle/preprocessing/select_initial_responses.py \
            --model_name $MODEL --dataset_name $BASE_DATASET
        
        conda activate llm-inference
        python src/dataset_creation/base_dataset_specific/isabelle/informal_to_formal/convert_to_formal.py \
            --base_model_name $MODEL --dataset_name $BASE_DATASET \
            --conversion_model_name $CONVERSION_MODEL
        conda deactivate
        
        python src/dataset_creation/base_dataset_specific/isabelle/error_detection/preprocessing.py \
            --base_model_name $MODEL --dataset_name $BASE_DATASET \
            --conversion_model_name $CONVERSION_MODEL \
            --num_parallel_processes 64
    done
done
