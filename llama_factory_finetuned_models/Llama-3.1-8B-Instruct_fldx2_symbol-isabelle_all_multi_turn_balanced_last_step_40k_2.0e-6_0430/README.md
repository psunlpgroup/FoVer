---
library_name: transformers
license: other
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Llama-3.1-8B-Instruct_fldx2_symbol-isabelle_all_multi_turn_balanced_last_step_40k_2.0e-6

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the FoVer_PRM_FormalLogic-FormalProof_balanced_last_step_40k_Llama-3.1-8B-Instruct dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-06
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- total_eval_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.5
- num_epochs: 1.0

### Training results



### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0
