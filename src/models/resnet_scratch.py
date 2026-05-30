#!/usr/bin/env python3
"""
Model 2: ResNet-style from scratch — Learning residual connections.

WHY RESNET:
  The key insight: deeper ≠ better without skip connections. A 20-layer
  plain CNN can outperform a 56-layer one because gradients vanish in
  deep networks. ResNet solves this with residual connections:

    output = F(x) + x    (instead of output = F(x))

  The network only needs to learn the RESIDUAL F(x) = output - x.
  If the optimal transformation is close to identity (common in deep
  networks where later layers refine rather than transform), learning
  a near-zero residual is much easier than learning the identity mapping.

WHY FROM SCRATCH (not pretrained):
  To demonstrate understanding of the architecture and to measure how
  much pretrained weights actually help. If a scratch ResNet-18 gets 85%
  and pretrained gets 95%, we can quantify the value of ImageNet pretraining
  for medical images.

ARCHITECTURE (ResNet-18 style):
  Initial conv + maxpool → 4 stages of 2 residual blocks each → GAP → FC

  Stage 1: 64 filters,  2 blocks, no downsampling
  Stage 2: 128 filters, 2 blocks, stride-2 downsampling
  Stage 3: 256 filters, 2 blocks, stride-2 downsampling
  Stage 4: 512 filters, 2 blocks, stride-2 downsampling

PARAMETER COUNT: ~11.2M (similar to official ResNet-18)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Basic residual block (2 convolutions + skip connection).

    Architecture:
      x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+x) → ReLU
                                                   ↑
                                              skip connection

    When dimensions change (downsampling or channel increase):
      The skip connection uses a 1×1 conv to match dimensions:
      x → Conv1x1(stride=2) → BN → (+F(x)) → ReLU

    Why 1×1 conv for the shortcut (not zero-padding or average pooling):
      - Zero-padding: introduces zeros that don't carry information
      - Average pooling: loses spatial detail
      - 1×1 conv: learnable projection, best accuracy
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection — identity if dimensions match, 1×1 conv otherwise
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # The critical line: add the input back
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNetScratch(nn.Module):
    """
    ResNet-18-style architecture built from scratch.

    Input:  (B, 3, 224, 224)
    Stem:   (B, 64, 56, 56)    — 7×7 conv stride 2 + maxpool
    Stage1: (B, 64, 56, 56)    — 2 residual blocks, no downsampling
    Stage2: (B, 128, 28, 28)   — 2 blocks, stride-2 on first
    Stage3: (B, 256, 14, 14)   — 2 blocks, stride-2 on first
    Stage4: (B, 512, 7, 7)     — 2 blocks, stride-2 on first
    GAP:    (B, 512)
    FC:     (B, 4)
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_blocks: list = None,
        initial_filters: int = 64,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]  # ResNet-18 config

        self.in_channels = initial_filters

        # Stem: aggressive downsampling to reduce computation
        # 224×224 → 112×112 (conv stride 2) → 56×56 (maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_filters, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 stages of residual blocks
        self.stage1 = self._make_stage(initial_filters, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(initial_filters * 2, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(initial_filters * 4, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(initial_filters * 8, num_blocks[3], stride=2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(initial_filters * 8, num_classes),
        )

        self._initialize_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build a stage of residual blocks.

        First block may downsample (stride=2), rest maintain dimensions.
        This is how ResNet progressively reduces spatial size while
        increasing channel depth.
        """
        blocks = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        """Return last conv layer for Grad-CAM (last block's conv2)."""
        last_block = list(self.stage4.children())[-1]
        return last_block.conv2
