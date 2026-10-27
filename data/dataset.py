import torch
from .data import img_read
from ..utils import json_load
from torch.utils.data import Dataset


# 简单示例
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = list(json_load(data_path).values())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["img_path"]["adc"]
        img = img_read(img_path,(4,1,1))

        target = item["text_label"]["modal_wise_finding"]["adc"]

        # img = (img - img.mean())/img.std()
        img = (img - 0.3141)/0.2623
        img = torch.tensor(img).unsqueeze(0)
        prompt = "This is an ADC map from a brain MRI. Please describe and analyze it."
        return img, prompt, target