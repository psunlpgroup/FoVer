source $CONDA_SH
conda activate llm-inference

# evaluate existing prms

python run/04_downstream_evaluation/run_sample_and_rank_sota_models.py

python src/downstream_evaluation/sample_and_rank/get_performance_and_table.py

conda deactivate
