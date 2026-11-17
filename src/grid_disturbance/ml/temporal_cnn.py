"""1D Temporal Convolutional Neural Network for raw time-series windows."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCNNBlock(nn.Module):
    """Convolutional block: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.relu(self.bn(self.conv(x)))))


class TemporalCNN(nn.Module):
    """1D Temporal Convolutional Neural Network for power grid disturbance classification."""

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        channel_sizes: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        dropout: float = 0.30,
    ):
        super().__init__()
        channels = channel_sizes or [32, 64, 128, 256]
        kernels = kernel_sizes or [7, 5, 3, 3]
        pools = pool_sizes or [2, 2, 2, 2]

        blocks = []
        curr_in = in_channels
        for out_c, k, p in zip(channels, kernels, pools, strict=False):
            blocks.append(
                TemporalCNNBlock(curr_in, out_c, kernel_size=k, pool_size=p, dropout=dropout)
            )
            curr_in = out_c

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, in_channels, seq_len)
        feat = self.features(x)
        pooled = self.global_pool(feat).squeeze(-1)
        logits = self.classifier(pooled)
        return logits
