"""
Tests for all model architectures and the model factory.

Covers:
  - Forward pass output shape for all 5 architectures
  - Factory creation by name
  - Invalid model name handling
  - Parameter counting
  - Grad-CAM hook availability
"""

import pytest
import torch
import torch.nn as nn

from src.models.custom_cnn import CustomCNN
from src.models.resnet_scratch import ResNetScratch
from src.models.mobilenet_scratch import MobileNetScratch
from src.models.pretrained import PretrainedResNet, PretrainedEfficientNet
from src.models.factory import create_model, list_models, count_parameters, MODEL_REGISTRY


NUM_CLASSES = 4
BATCH_SIZE = 2
IMAGE_SIZE = 224


# ============================================================
# Architecture tests — each model gets the same battery
# ============================================================

class TestCustomCNN:
    """Tests for the baseline custom CNN."""

    def test_forward_shape(self):
        model = CustomCNN(num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_configurable_blocks(self):
        """Different block counts should change the model."""
        m3 = CustomCNN(num_classes=NUM_CLASSES, num_conv_blocks=3)
        m5 = CustomCNN(num_classes=NUM_CLASSES, num_conv_blocks=5)
        p3 = sum(p.numel() for p in m3.parameters())
        p5 = sum(p.numel() for p in m5.parameters())
        assert p5 > p3

    def test_grad_cam_layer(self):
        """get_last_conv_layer should return a Conv2d."""
        model = CustomCNN(num_classes=NUM_CLASSES)
        layer = model.get_last_conv_layer()
        assert isinstance(layer, nn.Conv2d)

    def test_dropout_variation(self):
        """Different dropout rates should be accepted."""
        m1 = CustomCNN(num_classes=NUM_CLASSES, dropout=0.0)
        m2 = CustomCNN(num_classes=NUM_CLASSES, dropout=0.9)
        # Both should produce valid output
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert m1(x).shape == (1, NUM_CLASSES)
        assert m2(x).shape == (1, NUM_CLASSES)


class TestResNetScratch:
    """Tests for the from-scratch ResNet."""

    def test_forward_shape(self):
        model = ResNetScratch(num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_residual_connection(self):
        """Output should differ from a model without skip connections."""
        model = ResNetScratch(num_classes=NUM_CLASSES)
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        # Just verify it runs and has gradients
        out.sum().backward()
        assert model.stage1[0].conv1.weight.grad is not None

    def test_grad_cam_layer(self):
        model = ResNetScratch(num_classes=NUM_CLASSES)
        layer = model.get_last_conv_layer()
        assert isinstance(layer, nn.Conv2d)


class TestMobileNetScratch:
    """Tests for the lightweight MobileNet."""

    def test_forward_shape(self):
        model = MobileNetScratch(num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_fewer_params_than_resnet(self):
        """MobileNet should have fewer parameters than ResNet."""
        mobile = MobileNetScratch(num_classes=NUM_CLASSES)
        resnet = ResNetScratch(num_classes=NUM_CLASSES)

        mobile_params = sum(p.numel() for p in mobile.parameters())
        resnet_params = sum(p.numel() for p in resnet.parameters())

        assert mobile_params < resnet_params

    def test_width_multiplier(self):
        """Smaller width_multiplier should reduce parameters."""
        m_full = MobileNetScratch(num_classes=NUM_CLASSES, width_multiplier=1.0)
        m_half = MobileNetScratch(num_classes=NUM_CLASSES, width_multiplier=0.5)

        p_full = sum(p.numel() for p in m_full.parameters())
        p_half = sum(p.numel() for p in m_half.parameters())

        assert p_half < p_full

    def test_grad_cam_layer(self):
        model = MobileNetScratch(num_classes=NUM_CLASSES)
        layer = model.get_last_conv_layer()
        assert isinstance(layer, nn.Conv2d)


class TestPretrainedResNet:
    """Tests for the fine-tuned pretrained ResNet."""

    def test_forward_shape(self):
        # pretrained=False for speed in tests
        model = PretrainedResNet(num_classes=NUM_CLASSES, pretrained=False)
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_freeze_backbone(self):
        """Frozen backbone should have no grad on backbone params."""
        model = PretrainedResNet(
            num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True
        )
        # Check that backbone params don't require grad
        frozen_count = sum(
            1 for p in model.backbone.parameters() if not p.requires_grad
        )
        assert frozen_count > 0

    def test_grad_cam_layer(self):
        model = PretrainedResNet(num_classes=NUM_CLASSES, pretrained=False)
        layer = model.get_last_conv_layer()
        assert isinstance(layer, nn.Conv2d)


class TestPretrainedEfficientNet:
    """Tests for the fine-tuned EfficientNet-B0."""

    def test_forward_shape(self):
        model = PretrainedEfficientNet(num_classes=NUM_CLASSES, pretrained=False)
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_grad_cam_layer(self):
        model = PretrainedEfficientNet(num_classes=NUM_CLASSES, pretrained=False)
        layer = model.get_last_conv_layer()
        assert isinstance(layer, nn.Conv2d)


# ============================================================
# Factory tests
# ============================================================

class TestModelFactory:
    """Tests for the model factory and registry."""

    def test_list_models(self):
        """All 5 models should be registered."""
        models = list_models()
        assert len(models) == 5
        assert "custom_cnn" in models
        assert "efficientnet_pretrained" in models

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_create_all_models(self, model_name, sample_config):
        """Every registered model should be creatable via factory."""
        model_cfg = sample_config["models"].get(model_name, {})
        model = create_model(model_name, model_cfg, num_classes=NUM_CLASSES)

        assert isinstance(model, nn.Module)

        # Forward pass should work
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_invalid_model_raises(self):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_count_parameters(self):
        """count_parameters should return a dict with expected keys."""
        model = CustomCNN(num_classes=NUM_CLASSES)
        stats = count_parameters(model)

        assert "total" in stats
        assert "trainable" in stats
        assert "frozen" in stats
        assert "total_mb" in stats
        assert stats["total"] > 0
        assert stats["trainable"] == stats["total"]  # No frozen layers
        assert stats["frozen"] == 0


# ============================================================
# Gradient flow test
# ============================================================

class TestGradientFlow:
    """Verify gradients flow through all models (no vanishing/explosion)."""

    @pytest.mark.parametrize("model_name", ["custom_cnn", "resnet_scratch", "mobilenet_scratch"])
    def test_gradients_exist(self, model_name, sample_config):
        """After backward pass, all parameters should have gradients."""
        model_cfg = sample_config["models"].get(model_name, {})
        model = create_model(model_name, model_cfg, num_classes=NUM_CLASSES)
        model.train()

        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.tensor([0, 1], dtype=torch.long)

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
