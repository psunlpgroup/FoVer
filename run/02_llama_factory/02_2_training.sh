cd ../LLaMA-Factory-FoVer
conda activate llama_factory_fover

llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_202512_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_202512_1.0e-6.yaml
