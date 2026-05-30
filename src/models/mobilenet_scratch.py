#!/usr/bin/env python3
"""
Model 3: MobileNet-style lightweight CNN — Efficiency-focused.

WHY MOBILENET:
  Standard convolutions are expensive. A 3×3 conv on 256 input channels
  producing 512 output channels needs 256 × 512 × 3 × 3 = 1,179,648
  multiply-adds PER PIXEL. MobileNet's depthwise separable convolutions
  split this into two cheaper operations:

  Standard Conv:     in_ch × out_ch × k × k = 256 × 512 × 9 = 1,179,648
  Depthwise + Point: in_ch × k × k + in_ch × out_ch = 256×9 + 256×512 = 133,376

  That's ~9x fewer operations. The accuracy loss is small (1-3%) because
  most of the "work" in a convolution is channel mixing, not spatial filtering.

WHY FROM SCRATCH:
  Same ablation logic as ResNet — measure the architecture's capability
  without pretrained weights. Also demonstrates understanding of
  depthwise separable convolutions for the interview.

DEPTHWISE SEPARABLE CONVOLUTION EXPLAINED:
  Step 1 — Depthwise Conv: apply ONE 3×3 filter per input channel independently.
    - Input: (B, C_in, H, W)
    - Each channel gets its own 3×3 filter (groups=C_in)
    - Output: (B, C_in, H, W) — spatial filtering, no channel mixing
    - Parameters: C_in × 3 × 3

  Step 2 — Pointwise Conv: 1×1 conv to mix channels.
    - Input: (B, C_in, H, W)
    - 1×1 conv: C_in → C_out
    - Output: (B, C_out, H, W) — channel mixing, no spatial filtering
    - Parameters: C_in × C_out

WIDTH MULTIPLIER:
  Scales all channel counts by a factor (default 1.0).
  width_multiplier=0.5 → half the channels → ~4x fewer operations.
  Useful for deploying on edge devices or mobile phones.

PARAMETER COUNT: ~3.2M (at width_multiplier=1.0)
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block.

    Depthwise Conv → BN → ReLU → Pointwise Conv → BN → ReLU

    The key trick is `groups=in_channels` in the depthwise conv.
    This tells PyTorch to apply each filter to only one input channel
    instead of all channels. Result: spatial filtering without channel mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        # Depthwise: groups=in_channels means each channel has its own filter
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Pointwise: 1×1 conv to mix channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetScratch(nn.Module):
    """
    MobileNet-V1-style architecture built from scratch.

    Architecture (at width_multiplier=1.0):
      Stem:        (B, 3, 224, 224) → (B, 32, 112, 112)    — standard conv
      DW Block 1:  (B, 32, 112, 112) → (B, 64, 112, 112)
      DW Block 2:  (B, 64, 112, 112) → (B, 128, 56, 56)    — stride 2
      DW Block 3:  (B, 128, 56, 56) → (B, 128, 56, 56)
      DW Block 4:  (B, 128, 56, 56) → (B, 256, 28, 28)     — stride 2
      DW Block 5:  (B, 256, 28, 28) → (B, 256, 28, 28)
      DW Block 6:  (B, 256, 28, 28) → (B, 512, 14, 14)     — stride 2
      DW Block 7-11: (B, 512, 14, 14) — 5 blocks at same size
      DW Block 12: (B, 512, 14, 14) → (B, 1024, 7, 7)      — stride 2
      DW Block 13: (B, 1024, 7, 7) → (B, 1024, 7, 7)
      GAP:         (B, 1024)
      FC:          (B, 4)
    """

    def __init__(
        self,
        num_classes: int = 4,
        width_multiplier: float = 1.0,
        dropout: float = 0.2,
        in_channels: int = 3,
    ):
        super().__init__()

        def _make_channels(c: int) -> int:
            """Apply width multiplier and round to nearest 8 (for hardware efficiency)."""
            return max(8, int(c * width_multiplier + 4) // 8 * 8)

        # Channel progression (standard MobileNet-V1)
        # (out_channels, stride) for each depthwise separable block
        block_config = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        # Stem: standard conv (not depthwise — too few input channels)
        stem_channels = _make_channels(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Depthwise separable blocks
        layers = []
        current_channels = stem_channels
        for out_ch, stride in block_config:
            out_ch = _make_channels(out_ch)
            layers.append(DepthwiseSeparableConv(current_channels, out_ch, stride))
            current_channels = out_ch

        self.features = nn.Sequential(*layers)
        self.final_channels = current_channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current_channels, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        """Return last conv layer for Grad-CAM (last block's pointwise conv)."""
        last_block = list(self.features.children())[-1]
        # The pointwise conv is the second Sequential inside the block
        return list(last_block.pointwise.children())[0]  # The Conv2d
