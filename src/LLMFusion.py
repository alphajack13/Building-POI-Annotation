import os

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class LLMFusion(nn.Module):
    def __init__(self, model_path, llm_tokenizer, pretrained_weights_path, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.llm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.tokenizer = llm_tokenizer
        self.d_model = self.llm_model.config.hidden_size
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=r".*layers\.(26|27)\.self_attn\.(q_proj|v_proj)$",
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.llm_model = get_peft_model(self.llm_model, self.lora_config)
        self.llm_model.print_trainable_parameters()
        self.load_pretrained_weights(pretrained_weights_path)
        self.freeze()

    def load_pretrained_weights(self, pretrained_weights_path):
        if not os.path.isfile(pretrained_weights_path):
            raise FileNotFoundError(f"Pretrained LLMFusion checkpoint not found: {pretrained_weights_path}")

        checkpoint = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

        llm_fusion_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("LLMFusion."):
                llm_fusion_state_dict[key[len("LLMFusion."):]] = value
            elif key.startswith("module.LLMFusion."):
                llm_fusion_state_dict[key[len("module.LLMFusion."):]] = value

        if not llm_fusion_state_dict:
            llm_fusion_state_dict = checkpoint

        own_keys = set(self.state_dict().keys())
        matched_keys = [key for key in llm_fusion_state_dict if key in own_keys]
        if not matched_keys:
            raise ValueError(f"No LLMFusion weights in checkpoint: {pretrained_weights_path}")

        missing_keys, unexpected_keys = self.load_state_dict(llm_fusion_state_dict, strict=False)
        print(f"Loaded frozen LLMFusion weights from {pretrained_weights_path}")
        print(f"Matched LLMFusion parameters: {len(matched_keys)}")
        if missing_keys:
            print(f"Missing LLMFusion parameters: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected LLMFusion parameters: {len(unexpected_keys)}")

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, llm_attention_mask, llm_input_ids):
        with torch.no_grad():
            base_embeddings = self.llm_model.embed_tokens(llm_input_ids)
            outputs = self.llm_model(inputs_embeds=base_embeddings, attention_mask=llm_attention_mask)
            batch_size = llm_attention_mask.size(0)
            last_token_indices = llm_attention_mask.sum(dim=1) - 1
            last_hidden_state = outputs.last_hidden_state
            last_token_vectors = last_hidden_state[
                torch.arange(batch_size, device=llm_attention_mask.device),
                last_token_indices,
            ].to(dtype=torch.float32)
            return last_token_vectors
