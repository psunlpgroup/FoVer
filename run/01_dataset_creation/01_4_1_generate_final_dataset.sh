# make the FoVer dataset
for BASE_DATASET in fldx2_symbol isabelle_all
do
    ###
    # ground truth

    if [[ $BASE_DATASET == fld* ]]; then
        # postprocess the error labels
        python src/dataset_creation/postprocessing/add_cot.py \
            --model_name ground_truth --dataset_name $BASE_DATASET --generation_seed 1

        python src/dataset_creation/postprocessing/randomly_merge_steps.py \
            --model_name ground_truth --dataset_name $BASE_DATASET --generation_seed 1

        python src/dataset_creation/postprocessing/add_non_reasoning_steps.py \
            --model_name ground_truth --dataset_name $BASE_DATASET --generation_seed 1
    fi

    # get models list
    if [[ $BASE_DATASET == fld* ]]; then
        MODELS_LIST="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
    elif [[ $BASE_DATASET == "isabelle_all" ]]; then
        MODELS_LIST="Qwen/Qwen2.5-7B-Instruct"
    else
        echo "Unknown dataset name: $BASE_DATASET"
        exit 1
    fi

    ###
    # models
    for MODEL in $MODELS_LIST
    do
        # if fld or isabelle in the dataset name
        if [[ $BASE_DATASET == fld* || $BASE_DATASET == isabelle* ]]; then
            # postprocess the error labels
            python src/dataset_creation/postprocessing/add_cot.py \
                --model_name $MODEL --dataset_name $BASE_DATASET --generation_seed selected

            python src/dataset_creation/postprocessing/randomly_merge_steps.py \
                --model_name $MODEL --dataset_name $BASE_DATASET

            python src/dataset_creation/postprocessing/add_non_reasoning_steps.py \
                --model_name $MODEL --dataset_name $BASE_DATASET
        fi

        # instance-level correct ratio
        if [[ $BASE_DATASET == fld* ]]; then
            INSTANCE_CORRECT_RATIO=0.30
        elif [[ $BASE_DATASET == isabelle* ]]; then
            INSTANCE_CORRECT_RATIO=0.30
        else
            INSTANCE_CORRECT_RATIO=-1  # we do not specify the ratio for non-fld datasets
        fi

        # make the final dataset
        for SUFFIX in "step_merged" "step_merged.with_non_reasoning_steps"
        do
        python src/dataset_creation/postprocessing/generate_final_dataset.py \
            --model_name $MODEL --dataset_name $BASE_DATASET \
            --instance_correct_ratio $INSTANCE_CORRECT_RATIO \
            --suffix $SUFFIX
        done
    done
done


# merge sampled datasets
python src/dataset_creation/postprocessing/merge_datasets_simple.py \
    --input_paths fover_dataset/fldx2_symbol_multi_turn_balanced_last_step_10k/Llama-3.1-8B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl \
                    fover_dataset/fldx2_symbol_multi_turn_balanced_last_step_10k/Qwen2.5-7B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl \
    --output_path fover_dataset/fldx2_symbol_multi_turn_balanced_last_step_20k/merged-Llama-3.1-8B-Instruct-Qwen2.5-7B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl

python src/dataset_creation/postprocessing/merge_datasets_simple.py \
    --input_paths fover_dataset/fldx2_symbol_multi_turn_balanced_last_step_20k/merged-Llama-3.1-8B-Instruct-Qwen2.5-7B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl \
                    fover_dataset/isabelle_all_multi_turn_balanced_last_step_20k/Qwen2.5-7B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl \
    --output_path fover_dataset/fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k/merged-Llama-3.1-8B-Instruct-Qwen2.5-7B-Instruct/train.step_merged.with_non_reasoning_steps.jsonl
