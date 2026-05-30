#!/usr/bin/env python3
"""
Model 1: Custom CNN — The baseline.

WHY BUILD A CUSTOM CNN:
  Every model comparison needs a baseline. This is the "what can we achieve
  with a simple architecture and no pretraining?" answer. If a pretrained
  model beats this by 2%, the pretraining barely matters. If it beats it by
  15%, pretraining is clearly worth the domain gap.

ARCHITECTURE:
  4 convolutional blocks, each: Conv → BatchNorm → ReLU → MaxPool
  Followed by: Global Average Pooling → Dropout → FC → 4 classes

  Why Global Average Pooling (GAP) instead of Flatten:
    - Flatten: 7×7×256 = 12,544 parameters per FC neuron → overfitting risk
    - GAP: averages each 7×7 feature map to a single number → 256-dim vector
    - Reduces parameters by ~50x with minimal accuracy loss
    - Also makes the model resolution-independent (works on any input size)

  Why BatchNorm after Conv (not before ReLU):
    - Convention: Conv → BN → ReLU (original BatchNorm paper order)
    - BN normalizes the pre-activation distribution, so ReLU sees consistent inputs
    - Stabilizes training, allows higher learning rates, acts as mild regularizer

PARAMETER COUNT:
  ~1.2M parameters — small enough to train on a laptop GPU in <10 minutes.
  Compare: ResNet-18 has 11.7M, EfficientNet-B0 has 5.3M.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Single convolutional block: Conv2d → BatchNorm → ReLU → MaxPool.

    The fundamental building block. Each block:
      - Doubles the spatial receptive field (via 3×3 conv)
      - Halves spatial dimensions (via 2×2 max pool)
      - Optionally increases filter count (captures more complex features)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Simple 4-block CNN for brain tumor classification.

    Input: (B, 3, 224, 224)
    After block 1: (B, 32, 112, 112)   — 3→32 filters, spatial halved
    After block 2: (B, 64, 56, 56)     — 32→64 filters
    After block 3: (B, 128, 28, 28)    — 64→128 filters
    After block 4: (B, 256, 14, 14)    — 128→256 filters
    After GAP:     (B, 256)            — spatial dimensions collapsed
    After FC:      (B, 4)             — class logits
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_conv_blocks: int = 4,
        initial_filters: int = 32,
        dropout: float = 0.5,
        in_channels: int = 3,
    ):
        super().__init__()

        # Build convolutional blocks with doubling filter counts
        blocks = []
        current_channels = in_channels
        for i in range(num_conv_blocks):
            out_channels = initial_filters * (2 ** i)  # 32, 64, 128, 256
            blocks.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels

        self.features = nn.Sequential(*blocks)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP: (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),             # (B, C, 1, 1) → (B, C)
            nn.Dropout(dropout),
            nn.Linear(current_channels, num_classes),
        )

        # Weight initialization — Kaiming (He) for ReLU networks
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming initialization for Conv and Linear layers.

        Why Kaiming (not Xavier):
          Xavier assumes linear activations. ReLU kills half the outputs
          (negative values → 0), so Xavier underestimates the needed variance.
          Kaiming accounts for ReLU by multiplying variance by 2.

        Why this matters:
          Bad initialization → activations explode or vanish in early layers
          → gradients are useless → training stalls or diverges.
        """
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
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        """Return the last conv layer for Grad-CAM."""
        # Walk backwards through features to find last Conv2d
        for block in reversed(list(self.features.children())):
            for layer in reversed(list(block.block.children())):
                if isinstance(layer, nn.Conv2d):
                    return layer
        raise RuntimeError("No Conv2d layer found")
