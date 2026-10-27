"""视觉编码器抽象基类。"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch
from torch import nn

class VisionEncoderConfig(BaseModel):
    """视觉编码器config必要字段"""
    name: str   # 视觉编码器注册名称
    args: dict  # 初始化参数


class VisionEncoder(nn.Module, ABC):
    """3D 视觉编码器抽象基类:(B, C, D, H, W) 体数据 → patch token 序列。

    子类通过构造参数 cfg(plain dict,即 ModelConfig.vision_kwargs)实例化,
    不直接依赖全局 config,避免循环导入;后续接入新结构只需继承本类并注册。
    """

    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.cfg: VisionEncoderConfig = cfg

    def freeze(self) -> None:
        """冻结全部参数。"""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """解冻全部参数。"""
        for p in self.parameters():
            p.requires_grad = True


    @abstractmethod
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向。

        Args:
            images: (B, C, D, H, W),已归一化到 [0,1] 的体数据
        Returns:
            (tokens, attn_mask):tokens (B, L, out_dim);attn_mask (B, L) 可为
            None(None = 全部 token 有效;带 padding 的编码器才需要 mask)。
        """