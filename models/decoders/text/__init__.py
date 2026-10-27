"""文本解码器子包:注册表 + 构建工厂。"""
from .base import TextDecoderConfig, TextDecoder
from .qwen3 import Qwen3

# 注册表:config.model.vision_encoder 的取值域;新增结构在此登记
TEXT_DECODERS = {
    "Qwen3-0.6B": Qwen3,
}


def build_text_decoder(decoder_cfg: TextDecoderConfig) -> TextDecoder:
    """按注册表键构建文本解码器。"""

    name = decoder_cfg.name
    if name not in TEXT_DECODERS:
        raise KeyError(f"未知文本解码器 '{name}',可选:{sorted(TEXT_DECODERS)}")
    return TEXT_DECODERS[name](decoder_cfg)