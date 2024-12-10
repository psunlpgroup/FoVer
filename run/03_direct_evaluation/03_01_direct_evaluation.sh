source $CONDA_SH
conda activate llm-inference

python run/03_direct_evaluation/03_01_direct_evaluation.py
python src/analysis/tables/generate_direct_evaluation_performance_tables.py

conda deactivate
