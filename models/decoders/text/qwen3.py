import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import TextDecoder,TextDecoderConfig
from peft import LoraConfig, get_peft_model


class Qwen3(TextDecoder):
    def __init__(self, cfg: TextDecoderConfig):
        super().__init__(cfg)
        init_args = self.cfg.args
        self.load_model(**init_args)
        self.setup_image_token(init_args.get("image_token",None))
        self.system_prompt = init_args.get("system_prompt",None)

    def load_model(
            self,
            model_dir, # 模型参数
            lora_config=None,
            **kwargs
            ):
        """载入预训练LLM与tokenlizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True)

        # lora:注入 LoRA 层后拆掉 PEFT 外壳,使 self.model 恒为裸 Qwen3ForCausalLM
        # (get_text_embed/forward 依赖 embed_tokens、model、lm_head 等裸模型属性路径)
        if lora_config is not None:
            peft_model = get_peft_model(self.model, LoraConfig(**lora_config))
            peft_model.print_trainable_parameters()
            self.model = peft_model.base_model.model

    def get_text_embed(self, input_ids):
        """文本embedding"""
        qwen = self.model
        text_embeds = qwen.model.embed_tokens(input_ids)
        return text_embeds

    def build_prefix_ids(self, prompts, num_image_tokens, image_token):
        tok = self.tokenizer
        all_ids = []
        for p in prompts:
            user_content = image_token * num_image_tokens + p
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})

            out = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors=None, return_dict=False)

            if isinstance(out, str):
                out = tok(out, add_special_tokens=False)["input_ids"]
                if isinstance(out[0], list):
                    out = out[0]
            elif hasattr(out, "tolist"):
                out = out.tolist()
            elif isinstance(out, list) and out and isinstance(out[0], str):
                flat = tok(out, add_special_tokens=False)["input_ids"]
                out = flat[0] if isinstance(flat[0], list) else flat

            all_ids.append(list(out))
        return all_ids

    def forward(self, input_ids, vis_embeds, attention_mask=None):
        B, L = input_ids.shape
         # 文本嵌入
        text_embeds = self.get_text_embed(input_ids)       # (B, L, 1024)

        # 视觉特征 (B, N, 1024)
        assert vis_embeds.shape[-1] == text_embeds.shape[-1], \
            f"视觉特征维度{vis_embeds.shape} 不匹配文本特征维度{text_embeds.shape}"
        vis_embeds = vis_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype)

        
        # placeholder 原位替换
        image_mask = (input_ids == self._image_pad_id)                    # (B, L)
        assert image_mask.sum().item() == vis_embeds.shape[0] * vis_embeds.shape[1], \
            f"占位符数量 {image_mask.sum().item()} != 视觉 token 数量 {vis_embeds.shape[0]*vis_embeds.shape[1]}"
        mask = image_mask.unsqueeze(-1).expand_as(text_embeds)
        text_embeds = text_embeds.masked_scatter(mask, vis_embeds.reshape(-1, vis_embeds.shape[-1]))

        # 送入 Qwen3 主体
        outputs = self.model.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        return logits



# if __name__=="__main__":
#     cfg = TextDecoderConfig(**{
#             "name": "Qwen3-0.6B",
#             "args":{
#                 "model_dir": "/hpc/jtia475/model_weights/Qwen3-0.6B"
#             }
#         })

#     model = Qwen3(cfg)
#     img_len = 16
#     pad_tokens = "<|image_pad|>" * img_len
#     prompts = [
#         f"<|im_start|>{pad_tokens}<|im_end|>描述这张图片",
#         f"描述这张图片<|im_start|>{pad_tokens}<|im_end|>"
#     ]
#     text_feature = model.tokenlize(prompts)
#     ids,att_mask = text_feature.input_ids, text_feature.attention_mask
#     vis_embeds = torch.randn((2,16,1024)).float()
#     result = model(ids,vis_embeds,att_mask)
#     print(result)


