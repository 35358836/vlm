from pydantic import BaseModel
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# 默认图像embedding占位token
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"



class TextDecoderConfig(BaseModel):
    """文本解码器config必要字段"""
    name: str   # 视觉编码器注册名称
    args: dict  # 初始化参数




class TextDecoder(nn.Module, ABC):
    """
    语言解码器抽象基类。

    约定：
      - 视觉特征由外部(encoder + projector)准备好，形状 (B, N_img, hidden_dim)
      - 输入 input_ids 中已含 N_img 个 <|image_pad|> 占位符
      - 子类负责：加载模型、扩展词表、暴露 embed_tokens / 主体 / lm_head
    """

    def __init__(self, cfg:TextDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = None
        self._image_pad_id = None
        
        self.model = None

    # ---------- 子类必须实现 ----------
    @abstractmethod
    def load_model(self, model_dir: str, lora_config=None, **kwargs):
        """加载预训练权重，设置 self.model(主体)和 self.tokenizer。
        可选 lora_config 用于配置lora。
        """
        ...

    @abstractmethod
    def get_text_embed(self, input_ids) -> nn.Module:
        """返回词嵌入层，用于把 input_ids 变成 (B, L, D)。"""
        ...

    @abstractmethod
    def forward(self, input_ids, vision_features, attention_mask=None):
        """前向传播逻辑"""
        ...


    # ---------- 通用实现 ----------
    @property
    def image_pad_id(self) -> int:
        if self._image_pad_id is None:
            raise RuntimeError("请先调用 setup_image_token()")
        return self._image_pad_id

    def setup_image_token(self, image_token=None):
        """在 tokenizer 中加入占位符并扩展词表。"""
        if image_token is None:
            image_token = DEFAULT_IMAGE_TOKEN

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [image_token]}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._image_pad_id = self.tokenizer.convert_tokens_to_ids(image_token)
        return self._image_pad_id

    def tokenlize(self, prompt):
        """文本转tokenid"""
        return self.tokenizer(prompt, return_tensors="pt")

    def build_prefix_ids(self, prompts, num_image_tokens, image_token):
        tok = self.tokenizer
        all_ids = []
        for p in prompts:
            user_content = image_token * num_image_tokens + p
            messages = [{"role": "user", "content": user_content}]
            out = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors=None, return_dict=False)

            # 兜底:str / list[str] / tensor 都转成 list[int]
            if isinstance(out, str):
                out = tok(out, add_special_tokens=False)["input_ids"]
                if isinstance(out[0], list):   # batch 形式 [[...]]
                    out = out[0]
            elif hasattr(out, "tolist"):
                out = out.tolist()
            elif isinstance(out, list) and out and isinstance(out[0], str):
                # list[str]（极少见）
                flat = tok(out, add_special_tokens=False)["input_ids"]
                out = flat[0] if isinstance(flat[0], list) else flat

            all_ids.append(list(out))
        return all_ids

    def build_target_ids(self, targets):
        tok = self.tokenizer
        out = []
        for t in targets:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            # ChatML: assistant 回复以 <|im_end|> 结束
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None and im_end != tok.unk_token_id:
                ids = ids + [im_end]
            else:
                ids = ids + [tok.eos_token_id]
            out.append(ids)
        return out

    def decode(self, out_ids, skip_special_tokens=True):
        """生成结果解码。子类可覆盖(比如去掉 ChatML 的 assistant 前缀)。"""
        return self.tokenizer.batch_decode(
            out_ids, skip_special_tokens=skip_special_tokens)