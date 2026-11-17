"""2D Convolutional Neural Network for spectrograms and wavelet scalograms."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramCNNBlock(nn.Module):
    """2D Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.relu(self.bn(self.conv(x)))))


class SpectrogramCNN(nn.Module):
    """2D CNN for time-frequency spectrogram and scalogram classification."""

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        channel_sizes: list[int] | None = None,
        dropout: float = 0.30,
    ):
        super().__init__()
        channels = channel_sizes or [32, 64, 128]

        blocks = []
        curr_in = in_channels
        for out_c in channels:
            blocks.append(
                SpectrogramCNNBlock(curr_in, out_c, kernel_size=3, pool_size=2, dropout=dropout)
            )
            curr_in = out_c

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, in_channels, H, W)
        feat = self.features(x)
        pooled = self.global_pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return logits
