import torch
import torch.nn as nn
from typing import Optional, List, Dict
import argparse


class Featurizer(nn.Module):
    def __init__(self, backbone: nn.Module, bottleneck_dim: int):
        self.backbone = backbone
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck = bottleneck

    def bottleneck_dim(self) -> int:
        return self.bottleneck_dim 

    def forward(self, x: torch.tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.bottleneck(f)
        return f

    def get_parameters(self, backbone_lr: Optional[float] = 0.1, bottleneck_lr: Optional[float] = 1.) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        
        params = [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.bottleneck.parameters(), "lr": bottleneck_lr},
        ]

        return params

        


