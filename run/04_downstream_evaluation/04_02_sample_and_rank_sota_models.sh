source $CONDA_SH
conda activate llm-inference

# evaluate existing prms

python run/04_downstream_evaluation/run_sample_and_rank_sota_models.py --sample_k 7
python run/04_downstream_evaluation/run_sample_and_rank_sota_models.py --sample_k 7 --evaluate_skywork  # run this only when vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B is running in another terminal

python src/downstream_evaluation/sample_and_rank/get_performance_and_table.py

conda deactivate
