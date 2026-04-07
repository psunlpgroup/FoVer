source $CONDA_SH
conda activate llm-inference

# fldx2_symbol gsm8k bigmath_math_word_problems metamathqa_gsm8k
export BASE_DATASET="fldx2_symbol"

# initial responses for the FoVer dataset
if [ $BASE_DATASET == "fldx2_symbol" ]
then
    MODELS_LIST="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
    NUM_SAMPLES=3
    MAX_TOKENS=256
else
    # math theorem proving datasets
    MODELS_LIST="meta-llama/Llama-3.1-8B-Instruct"
    NUM_SAMPLES=1
    MAX_TOKENS=2048
fi

for MODEL in $MODELS_LIST
do
    for SPLIT in validation test train
    do

    if [ $BASE_DATASET == "fldx2_symbol" ]
    then
        # outputs from LLMs on the original dataset
        python src/dataset_creation/initial_answer_generation/generate_initial_answers.py \
            --model_name $MODEL --dataset_name $BASE_DATASET --split $SPLIT \
            --max_tokens $MAX_TOKENS --num_samples $NUM_SAMPLES
    else
        # we abuse the code for best-of-k sampling to generate initial responses for dataset creation
        python src/downstream_evaluation/sample_and_rank/generate_initial_responses.py \
            --model_name $MODEL --dataset_name $BASE_DATASET --split $SPLIT \
            --max_tokens $MAX_TOKENS --sample_k $NUM_SAMPLES \
            --prompt_type few-shot \
            --temperature 0.5 --top_k 40 \
            --generating_responses_for_dataset_creation \
            --updated_save_directory_for_response_generation_prompts "model_inputs/dataset_creation/initial_responses/${BASE_DATASET}" \
            --updated_save_directory_for_answer_extraction_prompts "model_inputs/dataset_creation/answer_extraction/${BASE_DATASET}"
    fi
    done
done

conda deactivate
