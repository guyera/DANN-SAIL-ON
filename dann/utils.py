import torch
import numpy as np

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(preds, dim=1)
    dataset_size = preds.size[0]
    return 100 * torch.sum(preds == labels)/dataset_size

def print_log(log: dict):
    for key in log.keys():
        print(f"{key}: {log[key]}")
