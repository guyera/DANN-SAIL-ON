import torch
import torch.nn as nn
from typing import Optional
from grl import GradientReverseLayer


class ClassifierHead(nn.Module):
    def __init__(self, head: Optional[nn.Module], features_dim: int, num_classes: int, is_discriminator: Optional[bool] = False, grl: Optional[GradientReverseLayer] = None):
        self.features_dim = features_dim
        self.num_classes = num_classes
        self.is_discriminator = is_discriminator
        self.grl = grl
        if self.is_discriminator:
            assert self.grl is not None
        if head:
            self.head = head
        else:
            self.head = nn.Linear(self.features_dim, self.num_classes)
        

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        out = self.head(f)
        if self.is_discriminator:
            out = self.grl(out)
        return out