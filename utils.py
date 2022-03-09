import torch
import numpy as np
from typing import List, Optional
import torch.nn.functional as F
import torchvision
import os


# logging utils

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(preds, dim=1)
    dataset_size = preds.size()[0]
    return 100 * torch.sum(preds == labels).item()/dataset_size


def print_log(log: dict):
    for key in log.keys():
        print(f"{key}: {log[key]}")
