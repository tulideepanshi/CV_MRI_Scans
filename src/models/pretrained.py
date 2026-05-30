#!/usr/bin/env python3
"""
Models 4 & 5: Pretrained ResNet-18 and EfficientNet-B0 — Transfer learning.

WHY TRANSFER LEARNING FOR MEDICAL IMAGES:
  These models were pretrained on ImageNet (1.2M natural images, 1000 classes).
  Brain MRIs look nothing like dogs and cars. So why does this work?

  Because early layers learn UNIVERSAL features:
    Layer 1-2: edges, gradients, textures (useful for ANY image)
    Layer 3-5: shapes, patterns, contours (somewhat transferable)
    Later layers: object parts, scenes (ImageNet-specific, need retraining)

  For medical images, the edge/texture detectors transfer well. The model
  already knows how to find boundaries and intensity gradients — we just
  need to teach it what those patterns mean for tumor classification.

  Empirically, pretrained models consistently beat training from scratch
  on small medical datasets, even with the domain gap. The gap is:
    - ImageNet: natural color images, 3 channels with real color
    - Brain MRI: grayscale converted to 3-channel, low color diversity

FINE-TUNING STRATEGY:
  We do FULL fine-tuning (freeze_backbone=False). Alternatives:
    - Freeze all: only train the new classification head. Fast but limited.
    - Freeze early layers: good compromise, but requires knowing which layers.
    - Full fine-tune: update everything. Best accuracy on small datasets
      when combined with low learning rate and weight decay.

  We use a lower learning rate (0.001) and AdamW with weight decay (0.0001)
  to prevent the pretrained weights from being destroyed too quickly.

WHY THESE TWO MODELS:
  ResNet-18: classic, well-understood, 11.7M params. The "safe bet."
  EfficientNet-B0: modern, compound-scaled, 5.3M params.
    Achieves ResNet-50 accuracy with 5x fewer parameters.
    Uses squeeze-and-excitation blocks (channel attention).

Usage:
    from src.models.pretrained import PretrainedResNet, PretrainedEfficientNet

    model = PretrainedResNet(num_classes=4, pretrained=True)
    model = PretrainedEfficientNet(num_classes=4, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedResNet(nn.Module):
    """
    ResNet-18 with pretrained ImageNet weights and a new classification head.

    What we change:
      - Remove original fc layer (1000 classes → our 4 classes)
      - Add dropout before the new fc layer
      - Optionally freeze backbone (we don't by default)

    The original ResNet-18 fc layer: Linear(512, 1000)
    Our replacement:                 Dropout → Linear(512, 4)
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Load pretrained backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classification head
        # ResNet-18's fc input features = 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Even if backbone is frozen, the new fc is always trainable
        # (it was just created, so requires_grad=True by default)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        """Return last conv layer for Grad-CAM."""
        # ResNet-18's last conv is inside layer4[-1].conv2
        return self.backbone.layer4[-1].conv2

    def get_feature_extractor(self) -> nn.Module:
        """Return everything except the fc layer (for embedding extraction)."""
        return nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool,
            nn.Flatten(),
        )


class PretrainedEfficientNet(nn.Module):
    """
    EfficientNet-B0 with pretrained ImageNet weights.

    WHY EFFICIENTNET:
      Compound scaling: instead of making networks deeper OR wider OR
      higher resolution (like ResNet did), EfficientNet scales all three
      dimensions together using a compound coefficient.

      The result: EfficientNet-B0 (5.3M params) matches ResNet-50 (25.6M params)
      at ~5x lower compute. On our small dataset, fewer parameters also means
      less overfitting risk.

    SQUEEZE-AND-EXCITATION (SE) BLOCKS:
      EfficientNet uses SE blocks — a form of channel attention:
        1. Global average pool → (B, C, 1, 1) — "squeeze"
        2. FC → ReLU → FC → Sigmoid → (B, C, 1, 1) — "excitation"
        3. Multiply original features by these channel weights

      This lets the network learn "pay more attention to channel 42,
      less to channel 17" — adaptive feature recalibration.

    What we change:
      - Remove original classifier (1000 classes)
      - Add Dropout → Linear(1280, 4)
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Load pretrained backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        # EfficientNet-B0's classifier input = 1280
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        """
        Return last conv layer for Grad-CAM.

        EfficientNet-B0's feature extractor ends with:
          features[8] → Conv2d(320, 1280, 1×1) + BN + SiLU
        The Conv2d inside that block is what we want.
        """
        # features[-1] is the final conv block (head)
        last_block = self.backbone.features[-1]
        # It's a Sequential: Conv2d → BatchNorm → SiLU
        for layer in last_block.children():
            if isinstance(layer, nn.Conv2d):
                return layer
        # Fallback: return the whole block
        return last_block

    def get_feature_extractor(self) -> nn.Module:
        """Return feature extractor (everything before classifier)."""
        return nn.Sequential(
            self.backbone.features,
            self.backbone.avgpool,
            nn.Flatten(),
        )
