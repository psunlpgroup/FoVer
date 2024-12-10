source $CONDA_SH
conda activate llm-inference

# initial responses for the FoVer dataset
for BASE_DATASET in fldx2_symbol gsm8k bigmath_math_word_problems metamathqa_gsm8k
    do

    if [ $BASE_DATASET == "fldx2_symbol" ]
    then
        NUM_SAMPLES=3
        MAX_TOKENS=256
    else
        # math theorem proving datasets
        NUM_SAMPLES=3
        MAX_TOKENS=2048
    fi

    for MODEL in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
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
                --temperature 0.5 --top_k 40
        fi
        done
    done
done

conda deactivate
