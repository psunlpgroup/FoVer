import torch
import numpy as np
import transformers
import vllm
import vllm.config


class InferenceModel():
    def __init__(self, model_name: str, use_vllm: bool=True, use_vllm_reward_task: bool=False):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_hf_pipeline = False

        self.tokenizer = None  # load tokenizer later
        self.model: vllm.LLM | None = None  # load model later
        
        self.is_reward_model = use_vllm_reward_task

    def load_model(self):
        # Model is already loaded
        if self.model is not None:
            return True

        if self.model_name == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
            self.model = "This model should be used with vllm server. Refer to https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
            print(self.model)
            return
        
        # Load the model
        print(f"Loading model {self.model_name}...")\
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name
        )
        
        if self.use_vllm:
            num_gpus = torch.cuda.device_count()
            
            if self.is_reward_model:
                # reward models
                
                if self.model_name in [
                            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
                            "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
                        ]:
                    ###
                    # This part does not work
                    # import warnings
                    # warnings.warn(
                    #     "This code does not return proper rewards for RLHFlow models. "
                    #     "This code needs to be fixed in the future. "
                    #     "Refer to https://github.com/vllm-project/vllm/issues/16545"
                    # )
                    # RLHFlow model
                    # we need to update the pooling config
                    # https://docs.vllm.ai/en/latest/models/supported_models.html#reward-modeling-task-reward
                    # pooler_config = vllm.config.PoolerConfig(
                    #     pooling_type="STEP", step_tag_id=17165,
                    #     returned_token_ids=[12, 10]
                    # )
                    ###
                    
                    # This is a causal model, so we need to provide the
                    # step_tag_id for the previous token for the target step
                    # tag token. 271 = "\n\n" which is the end of the tags.
                    # We need to remove misdetected "\n\n" that are not from
                    # the tags
                    pooler_config = vllm.config.PoolerConfig(
                        pooling_type="STEP", # step_tag_id=271,
                    )
                    
                    self.model = vllm.LLM(
                        self.model_name, tensor_parallel_size=num_gpus,
                        task="reward",
                        override_pooler_config=pooler_config
                    )
                elif self.model_name in [
                            "Qwen/Qwen2.5-Math-7B-PRM800K",
                            "Qwen/Qwen2.5-Math-PRM-7B",
                            "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
                        ]:
                    # Qwen model
                    self.model = vllm.LLM(
                        self.model_name, tensor_parallel_size=num_gpus,
                        task="reward"
                    )
                else:
                    import warnings
                    warnings.warn(
                        f"Model {self.model_name} does not have a specific reward model setting implemented. "
                        "Using default settings for reward model. This may not work as expected."
                    )
                    
                    # This part does not work
                    
                    ###
                    # step_tag_id = self.tokenizer.encode("\u043a\u0438", add_special_tokens=False)[0]
                    # correct_id = self.tokenizer.encode("+", add_special_tokens=False)[0]
                    # incorrect_id = self.tokenizer.encode("-", add_special_tokens=False)[0]
                    # print(f"step_tag_id: {step_tag_id}, correct_id: {correct_id}, incorrect_id: {incorrect_id}")
                    
                    # pooler_config = vllm.config.PoolerConfig(
                    #     pooling_type="STEP",
                    #     step_tag_id=step_tag_id,
                    #     returned_token_ids=[incorrect_id, correct_id],
                    # )
                    ###
                    
                    # this code simply get all last hidden states
                    pooler_config = vllm.config.PoolerConfig(
                        pooling_type="STEP"
                    )
                    
                    self.model = vllm.LLM(
                        self.model_name, tensor_parallel_size=num_gpus,
                        task="reward",
                        override_pooler_config=pooler_config
                    )
            else:
                if "gemma-3" in self.model_name:
                    # standard models
                    self.model = vllm.LLM(
                        self.model_name, tensor_parallel_size=num_gpus,
                        dtype="bfloat16"
                        # this is not necesary in futuer version of vllm
                        # https://github.com/vllm-project/vllm/pull/17629
                    )
                else:
                    # standard models
                    self.model = vllm.LLM(
                        self.model_name, tensor_parallel_size=num_gpus
                    )
        else:
            self.use_hf_pipeline = True
            
            self.model = transformers.pipeline(
                "text-generation", model=self.model_name, device_map="auto"
            )


    def postprocess_reward_model_output(self, output: vllm.RequestOutput):
        """
        Retrieve features corresponding to the target tokens
        """

        if not self.is_reward_model:
            raise RuntimeError("This model is not a reward model")

        # some models do not need to be postprocessed
        if self.model_name in [
                    "Qwen/Qwen2.5-Math-7B-PRM800K",
                    "Qwen/Qwen2.5-Math-PRM-7B",
                ]:
            return output.outputs.data.tolist()
        elif self.model_name == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
            return output
        
        # other models
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")

        architectures_config = self.model.llm_engine.model_config.hf_text_config.architectures
        if len(self.model.llm_engine.model_config.hf_text_config.architectures) > 1:
            raise NotImplementedError(
                "This code only supports models with one architecture"
            )

        features: torch.Tensor = output.outputs.data
        tokenized_prompt = np.array(output.prompt_token_ids)

        from src.prm.postprocessing import get_step_token_position

        # # we get the feature for the last token in the tag for the assistant
        # # conversation. It includes the prediction for the next token (the
        # # first token in the assistant's response), which is the reward
        # target_id_position_candidates = []

        # # we use multiple tokens to detect the target token id because
        # # target tokens can be also included in other parts
        # if architectures_config[0] == "LlamaForCausalLM":
        #     # target token id
        #     target_token_id = self.tokenizer.encode(
        #         "\n\n", add_special_tokens=False)[0]
        #     target_id_position_candidates.append(
        #         tokenized_prompt == target_token_id
        #     )

        #     # assistant id
        #     assistant_token_id = self.tokenizer.encode(
        #         "assistant", add_special_tokens=False)[0]
        #     ids = np.where(
        #         tokenized_prompt == assistant_token_id
        #     )[0] + 2
        #     mask = np.zeros_like(tokenized_prompt, dtype=bool)
        #     mask[ids] = True
        #     target_id_position_candidates.append(mask)

        #     # <|end_header_id|>
        #     end_header_id = self.tokenizer.encode(
        #         "<|end_header_id|>", add_special_tokens=False)[0]
        #     ids = np.where(
        #         tokenized_prompt == end_header_id
        #     )[0] + 1
        #     mask = np.zeros_like(tokenized_prompt, dtype=bool)
        #     mask[ids] = True
        #     target_id_position_candidates.append(mask)

        # elif architectures_config[0] == "Qwen2ForCausalLM":
        #     # target token id
        #     target_token_id = self.tokenizer.encode(
        #         "\n", add_special_tokens=False)[0]
        #     target_id_position_candidates.append(
        #         tokenized_prompt == target_token_id
        #     )

        #     # assistant id
        #     assistant_token_id = self.tokenizer.encode(
        #         "assistant", add_special_tokens=False)[0]
        #     ids = np.where(
        #         tokenized_prompt == assistant_token_id
        #     )[0] + 1
        #     mask = np.zeros_like(tokenized_prompt, dtype=bool)
        #     mask[ids] = True
        #     target_id_position_candidates.append(mask)

        #     # <|im_start|>
        #     end_header_id = self.tokenizer.encode(
        #         "<|im_start|>", add_special_tokens=False)[0]
        #     ids = np.where(
        #         tokenized_prompt == end_header_id
        #     )[0] + 2
        #     mask = np.zeros_like(tokenized_prompt, dtype=bool)
        #     mask[ids] = True
        #     target_id_position_candidates.append(mask)

        # else:
        #     raise NotImplementedError(
        #         f"This code does not support {self.model_name} model"
        #     )

        # # take and
        # target_id_position = np.where(
        #     np.logical_and.reduce(target_id_position_candidates)
        # )[0]

        # # extract features
        # target_id_position = torch.from_numpy(target_id_position)
        
        target_id_position = get_step_token_position(
            tokenized_prompt=tokenized_prompt,
            tokenizer=self.tokenizer,
            model_type={
                "LlamaForCausalLM": "llama",
                "Qwen2ForCausalLM": "qwen",
            }[architectures_config[0]]
        )
        
        features = features[target_id_position]

        return features.tolist()
