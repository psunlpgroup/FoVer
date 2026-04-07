source $CONDA_SH
conda activate llm-inference

# pass@k (oracle verification)
python run/04_downstream_evaluation/run_sample_and_rank_oracle.py

# evaluate prms
SAMPLE_K=7
python run/04_downstream_evaluation/run_sample_and_rank.py --sample_k $SAMPLE_K
python src/downstream_evaluation/sample_and_rank/get_performance_and_table.py --add_bootstrap_test --sample_k $SAMPLE_K --selection_method max
python src/downstream_evaluation/sample_and_rank/get_performance_and_table.py --add_bootstrap_test --sample_k $SAMPLE_K --selection_method weighted_majority

conda deactivate
