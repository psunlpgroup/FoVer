source $CONDA_SH

###
# LLM Inference

conda env create -f src/llm/environments/inference_environment.yml
conda activate llm-inference

# huggingface
pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.50.3

MAX_JOBS=4 pip install flash-attn==2.7.2.post1 --no-build-isolation

# accelerate inference
pip install vllm==0.8.2

conda deactivate
