"""Compact MLP classifier for CIELAB-derived tabular color features.

Two variants, selected empirically via src/ablation.py rather than assumed:
- PlainMLP:    D -> hidden... -> K, optional BatchNorm + Dropout
- ResidualMLP: one residual block (h3 = h1 + h2) before the classifier head
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
}


def _act(name: str) -> nn.Module:
    return _ACTIVATIONS[name]()


class PlainMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], n_classes: int,
                 activation: str = "gelu", dropout: float = 0.1, use_batchnorm: bool = True):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


class ResidualMLP(nn.Module):
    """h1 = phi(W1 x + b1); h2 = phi(W2 h1 + b2); h3 = h1 + h2; logits = W_o h3 + b_o.
    hidden_dims must all be equal (required for the residual add)."""

    def __init__(self, in_dim: int, hidden_dims: list[int], n_classes: int,
                 activation: str = "gelu", dropout: float = 0.1, use_batchnorm: bool = True):
        super().__init__()
        assert len(hidden_dims) >= 2 and len(set(hidden_dims)) == 1, \
            "ResidualMLP requires >=2 equal-width hidden layers for the residual add"
        width = hidden_dims[0]
        self.in_proj = nn.Linear(in_dim, width)
        self.bn1 = nn.BatchNorm1d(width) if use_batchnorm else nn.Identity()
        self.act1 = _act(activation)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width) if use_batchnorm else nn.Identity()
        self.act2 = _act(activation)
        self.drop2 = nn.Dropout(dropout)

        extra = hidden_dims[2:]
        extra_layers = []
        prev = width
        for h in extra:
            extra_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h) if use_batchnorm else nn.Identity(),
                              _act(activation), nn.Dropout(dropout)]
            prev = h
        self.extra = nn.Sequential(*extra_layers)
        self.head = nn.Linear(prev, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.drop1(self.act1(self.bn1(self.in_proj(x))))
        h2 = self.drop2(self.act2(self.bn2(self.fc2(h1))))
        h3 = h1 + h2
        h3 = self.extra(h3)
        return self.head(h3)


def build_model(in_dim: int, n_classes: int, cfg: dict) -> nn.Module:
    kind = cfg.get("kind", "plain")
    hidden_dims = cfg.get("hidden_dims", [64, 32])
    activation = cfg.get("activation", "gelu")
    dropout = cfg.get("dropout", 0.1)
    use_batchnorm = cfg.get("batchnorm", True)
    if kind == "residual":
        return ResidualMLP(in_dim, hidden_dims, n_classes, activation, dropout, use_batchnorm)
    return PlainMLP(in_dim, hidden_dims, n_classes, activation, dropout, use_batchnorm)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
