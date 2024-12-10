# code for formal logic dataset creation

source $CONDA_SH
conda activate fover_data_creation

# generate error labels for the ground-truth steps (all steps should be correct)
python src/dataset_creation/base_dataset_specific/fol/dataset_generation/generate_error_labels.py \
    --model_name ground_truth --dataset_name fldx2_symbol

cd ../FLD-generator
# translation of symbols to text
python ../FoVer/src/dataset_creation/base_dataset_specific/fol/dataset_generation/translate_symbols_to_text.py \
    --model_name ground_truth --dataset_name fldx2_symbol --num_samples 1
cd ../FoVer

# automatic error annotation for the initial responses
for MODEL in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
do
    # automatically verify the outputs
    python src/dataset_creation/base_dataset_specific/fol/dataset_generation/generate_error_labels.py \
        --model_name $MODEL --dataset_name fldx2_symbol

    # cd ../FLD-generator
    # # translation of symbols to text
    # # this is not used in the paper
    # python ../FoVer/src/dataset_creation/base_dataset_specific/fol/dataset_generation/translate_symbols_to_text.py \
    #     --model_name $MODEL --dataset_name fldx2_symbol
    # cd ../FoVer

    for BASE_DATASET in fldx2_symbol # fidx2_text
    do
        # select challenging case from multiple initial responses
        python src/dataset_creation/postprocessing/select_from_multiple_initial_responses.py \
            --model_name $MODEL --dataset_name $BASE_DATASET
    done
done

conda deactivate
