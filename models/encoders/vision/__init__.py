"""视觉编码器子包:注册表 + 构建工厂。"""
from .base import VisionEncoderConfig, VisionEncoder
from .neurovfm import NeurovfmVisionEncoder

# 注册表:config.model.vision_encoder 的取值域;新增结构在此登记
VISION_ENCODERS = {
    "neurovfm": NeurovfmVisionEncoder,
}


def build_vision_encoder(encoder_cfg: VisionEncoderConfig) -> VisionEncoder:
    """按注册表键构建视觉编码器。"""

    name = encoder_cfg.name
    if name not in VISION_ENCODERS:
        raise KeyError(f"未知视觉编码器 '{name}',可选:{sorted(VISION_ENCODERS)}")
    return VISION_ENCODERS[name](encoder_cfg)
