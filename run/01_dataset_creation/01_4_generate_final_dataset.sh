MODELS_LIST="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"

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
        fi

        # if fld in the dataset name
        if [[ $BASE_DATASET == fld* ]]; then
            INSTANCE_CORRECT_RATIO=0.30
        elif [[ $BASE_DATASET == isabelle* ]]; then
            INSTANCE_CORRECT_RATIO=0.30
        else
            INSTANCE_CORRECT_RATIO=-1  # we do not specify the ratio for non-fld datasets
        fi

        # make the final dataset
        python src/dataset_creation/postprocessing/generate_final_dataset.py \
            --model_name $MODEL --dataset_name $BASE_DATASET \
            --instance_correct_ratio $INSTANCE_CORRECT_RATIO
    done
done


# merge sampled datasets
for MODEL in $MODELS_LIST
do
    # merge the datasets
    python src/dataset_creation/postprocessing/merge_datasets.py \
        --model_name $MODEL \
        --dataset_names_list fldx2_symbol isabelle_all
done


# generate dataset statistics
python src/dataset_creation/generate_dataset_statistics_table.py
