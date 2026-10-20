"""CNN kecil untuk mengklasifikasikan penyebab langsung peringatan deforestasi."""
from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DriverCNN(nn.Module):
    """Tiga blok konvolusi + global average pooling + classifier.

    Input  : (B, C, H, W) -- kanal Sentinel-1/2 + mask peringatan
    Output : (B, n_classes) logits

    Karena patch lebih besar dari area peringatan, jaringan melihat bentuk
    area yang berubah sekaligus konteks lingkungan di sekitarnya.
    """

    def __init__(self, in_channels: int, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),
            _block(width, width * 2),
            _block(width * 2, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))
