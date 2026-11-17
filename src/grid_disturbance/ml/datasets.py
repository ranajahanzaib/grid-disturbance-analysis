"""PyTorch Dataset classes for 1D signals and 2D time-frequency representations."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from grid_disturbance.signals.transforms import compute_spectrogram_matrix


class Signal1DDataset(Dataset):
    """PyTorch Dataset for 1D raw or filtered signal waveforms."""

    def __init__(
        self,
        signals: list[np.ndarray],
        labels: list[int],
        target_length: int = 600,
    ):
        self.signals = signals
        self.labels = labels
        self.target_length = target_length

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig = np.asarray(self.signals[idx], dtype=np.float32)

        # Pad or crop to target_length
        if len(sig) < self.target_length:
            pad = self.target_length - len(sig)
            sig = np.pad(sig, (0, pad), mode="constant")
        elif len(sig) > self.target_length:
            sig = sig[: self.target_length]

        # Normalize per-window
        std = float(np.std(sig))
        if std > 1e-6:
            sig = ((sig - np.mean(sig)) / std).astype(np.float32)

        # Add channel dimension: shape (1, target_length)
        tensor_x = torch.from_numpy(sig).unsqueeze(0)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for 2D time-frequency spectrogram representations."""

    def __init__(
        self,
        signals: list[np.ndarray],
        labels: list[int],
        sampling_rate: float = 30.0,
        image_size: tuple[int, int] = (64, 64),
    ):
        self.signals = signals
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig = np.asarray(self.signals[idx], dtype=np.float64)
        spec = compute_spectrogram_matrix(
            sig,
            sampling_rate=self.sampling_rate,
            output_shape=self.image_size,
        )
        # Add channel dimension: shape (1, H, W)
        tensor_x = torch.from_numpy(spec).unsqueeze(0)
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y
