# code for formal theorem proving dataset creation

for MODEL_NAME in "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct"
do

export CONVERSION_MODEL=$MODEL_NAME
export FORMAL_PROOF_MODEL=$MODEL_NAME

conda activate llm-inference

for BASE_DATASET in gsm8k metamathqa_gsm8k bigmath_math_word_problems
do
    for MODEL in $MODEL_NAME
    do
        conda activate llm-inference

        # create formal theorem 
        python src/dataset_creation/base_dataset_specific/isabelle/informal_to_formal/generate_statement_and_proof.py \
            --base_model_name $MODEL --dataset_name $BASE_DATASET \
            --conversion_model_name $CONVERSION_MODEL \
            --formal_proof_generation_model_name $FORMAL_PROOF_MODEL

        conda deactivate
        ###
        
        python src/dataset_creation/base_dataset_specific/isabelle/error_detection/preprocessing.py \
            --base_model_name $MODEL --dataset_name $BASE_DATASET \
            --conversion_model_name $CONVERSION_MODEL \
            --formal_proof_generation_model_name $FORMAL_PROOF_MODEL \
            --num_parallel_processes 64
    done
done

done
