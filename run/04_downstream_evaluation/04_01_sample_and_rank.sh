source $CONDA_SH
conda activate llm-inference

# pass@k (oracle verification)
python run/04_downstream_evaluation/run_sample_and_rank_oracle.py

# evaluate prms
python run/04_downstream_evaluation/run_sample_and_rank.py
python src/downstream_evaluation/sample_and_rank/get_performance_and_table.py

conda deactivate
