"""
models/physnet.py
==================
What this file does:
  - Implements PhysNet architecture (Chen & McDuff, 2018)
  - 3D CNN that takes raw video patches as input
  - Outputs a continuous PPG waveform
  - Better than FFT-only approach for non-stationary HR

Interview explanation:
  "PhysNet is a 3D convolutional network — it applies convolutions
   across both spatial (x,y) and temporal (t) dimensions simultaneously.
   This is important because the rPPG signal is a spatio-temporal
   pattern: it's not just about one pixel over time, but about how
   groups of nearby pixels change together. A 1D temporal model
   would miss the spatial correlation; a 2D spatial model applied
   per-frame would miss the temporal structure. 3D conv captures both."

Architecture:
  Input:  (batch, 3, T, H, W) — 3 channels, T frames, H×W spatial
  Output: (batch, T) — predicted PPG waveform at each timestep

  5 stages of Conv3D + BatchNorm + ReLU + Pooling
  Final: average pooling across spatial dims → 1D signal per timestep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PhysNetBlock(nn.Module):
    """
    One convolutional block: Conv3D → BatchNorm → ReLU.
    Used as the building block of PhysNet.

    Why BatchNorm?
      Normalizes activations within each mini-batch.
      Stabilizes training and allows higher learning rates.
      Especially important here because the input pixel values
      vary widely across different lighting conditions.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int],
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False   # bias=False when using BatchNorm
        )
        self.bn   = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class PhysNet(nn.Module):
    """
    PhysNet: End-to-end rPPG estimation via 3D CNN.
    Reference: Chen & McDuff (2018), NeurIPS Workshop.

    Key design decisions:
      1. 3D convolutions: capture spatio-temporal skin color patterns
      2. Progressive spatial downsampling: reduce computation while
         preserving temporal resolution (we need temporal detail for HRV)
      3. No temporal downsampling in early layers: preserve all heartbeat
         timing information until final pooling
      4. Sigmoid output: PPG waveform values mapped to 0–1 range

    Input dimensions:
      T = 128 frames (~4.3 seconds at 30fps) per clip
      H = W = 32 pixels (forehead patch, downsampled for efficiency)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Stage 1: initial feature extraction
        # Large temporal kernel (3) to capture multi-frame patterns
        self.stage1 = nn.Sequential(
            PhysNetBlock(in_channels, 32,
                         kernel_size=(1, 5, 5),
                         padding=(0, 2, 2)),
            PhysNetBlock(32, 64,
                         kernel_size=(3, 3, 3),
                         padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2),
                         stride=(1, 2, 2))    # spatial ↓2, temporal preserved
        )

        # Stage 2: deeper temporal patterns
        self.stage2 = nn.Sequential(
            PhysNetBlock(64, 64,
                         kernel_size=(3, 3, 3),
                         padding=(1, 1, 1)),
            PhysNetBlock(64, 64,
                         kernel_size=(3, 3, 3),
                         padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2),
                         stride=(1, 2, 2))    # spatial ↓2, temporal preserved
        )

        # Stage 3: compress spatial, preserve temporal
        self.stage3 = nn.Sequential(
            PhysNetBlock(64, 64,
                         kernel_size=(3, 3, 3),
                         padding=(1, 1, 1)),
            PhysNetBlock(64, 64,
                         kernel_size=(3, 3, 3),
                         padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2),
                         stride=(1, 2, 2))    # spatial ↓2, temporal preserved
        )

        # Stage 4: near-final temporal features
        self.stage4 = nn.Sequential(
            PhysNetBlock(64, 64,
                         kernel_size=(3, 3, 3),
                         padding=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((None, 2, 2))  # spatial → 2×2, temporal free
        )

        # Stage 5: collapse spatial completely → 1D temporal signal
        self.stage5 = nn.Sequential(
            nn.Conv3d(64, 1,
                      kernel_size=(1, 2, 2),
                      stride=(1, 1, 1),
                      padding=0,
                      bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for conv layers.
        Why Xavier? It keeps activation variances stable across layers,
        preventing vanishing/exploding gradients in deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
          x: (batch, 3, T, H, W)
             batch = number of clips
             3 = RGB channels
             T = number of frames (e.g. 128)
             H, W = spatial dimensions (e.g. 32×32)

        Returns:
          ppg: (batch, T) — predicted PPG waveform
        """
        x = self.stage1(x)   # (B, 64, T, H/2, W/2)
        x = self.stage2(x)   # (B, 64, T, H/4, W/4)
        x = self.stage3(x)   # (B, 64, T, H/8, W/8)
        x = self.stage4(x)   # (B, 64, T, 2, 2)
        x = self.stage5(x)   # (B, 1, T, 1, 1)

        # Remove spatial and channel dims → (B, T)
        ppg = x.squeeze(-1).squeeze(-1).squeeze(1)

        return ppg


def get_physnet_loss(pred: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
    """
    Negative Pearson correlation loss.

    Why Pearson and not MSE?
      The absolute amplitude of the rPPG signal varies between
      subjects and sessions (different skin tones, lighting).
      We don't care about amplitude — we care about the SHAPE
      (timing of peaks = heart rate timing).
      Pearson correlation is amplitude-invariant: it measures
      how well the predicted waveform's shape matches the target.

    Range: -1 (perfect anti-correlation) to +1 (perfect correlation)
    We minimize the negative correlation (= maximize correlation).

    Args:
      pred:   (batch, T) predicted PPG
      target: (batch, T) ground truth PPG

    Returns:
      loss scalar
    """
    pred_mean   = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_std   = pred.std(dim=1, keepdim=True)   + 1e-8
    target_std = target.std(dim=1, keepdim=True) + 1e-8

    pred_norm   = (pred   - pred_mean)   / pred_std
    target_norm = (target - target_mean) / target_std

    pearson = (pred_norm * target_norm).mean(dim=1)

    # Average over batch, minimize negative correlation
    loss = -pearson.mean()
    return loss
