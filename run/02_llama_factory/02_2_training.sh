cd ../LlaMa-Factory-FoVer
conda activate llama_factory_fover

###
# main training

# we select the model with the best performance on the validation set

llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-5.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-5.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-5.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol_multi_turn_balanced_last_step_20k_1.0e-5.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_isabelle_all_multi_turn_balanced_last_step_20k_1.0e-5.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_1.0e-5.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

###
# ablation study

# dataset size
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k_5.0e-6.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_5k_duplicated_40k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_10k_duplicated_40k_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_duplicated_40k_2.0e-6.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

# label inbalance
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25_5.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75_5.0e-6.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*

llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.25_2.0e-6.yaml
llamafactory-cli train ../FoVer/llama_factory_config/Qwen2.5-7B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_20k_correct=0.75_2.0e-6.yaml

rm -r ../FoVer/llama_factory_finetuned_models/**/checkpoint-*
