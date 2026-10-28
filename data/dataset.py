import torch
from typing import NamedTuple
from .data import load_to_tokens
from ..utils import json_load
from torch.utils.data import Dataset


# 一个 batch 的视觉输入:各样本变长 token 已右侧零填充对齐
class VisionBatch(NamedTuple):
    tokens: torch.Tensor  # (B, N, patch_feat)
    coords: torch.Tensor  # (B, N, 3)
    mask: torch.Tensor    # (B, N) 有效位(背景 token 已在数据侧去除)

    @property
    def shape(self):  # 兼容 img.shape[0] 风格的 batch 大小取用
        return (self.tokens.shape[0],)

    def size(self, dim=None):  # 兼容 img.size(0) 风格
        return self.shape if dim is None else self.shape[dim]


# 简单示例
class MyDataset(Dataset):
    def __init__(self, data_path, modality="mri"):
        self.data = []
        for item in json_load(data_path).values():
            label_map, img_map = item["text_label"]["modal_wise_finding"], item["img_path"]
            for k in label_map:
                self.data.append({
                    "text_label": label_map[k],
                    "img_path": img_map[k],
                    "modality": k
                })

        self.modality = modality

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["img_path"]
        # 编码器官方预处理(RPI 定向、1x1x4mm 重采样、归一化、去背景)→ 变长 token 序列
        tokens, coords = load_to_tokens(img_path, modality=self.modality)

        target = item["text_label"]
        prompt = f"This is a {item['modality']} map from a brain MRI. Please describe and analyze it."
        return tokens, coords, prompt, target
