source $CONDA_SH
conda activate llm-inference

# run evaluation for model selection (hyperparameter search)
python run/04_downstream_evaluation/run_sample_and_rank.py --evaluation_mode model_selection

# get performance table and graph
python src/downstream_evaluation/sample_and_rank/get_performance_and_table.py --evaluation_mode model_selection

conda deactivate
