"""
Integration and regression tests.

These tests verify end-to-end flows rather than individual units:
  - Full inference pipeline: image → preprocess → model → prediction
  - Model save/load round-trip (checkpoint integrity)
  - Config-driven model creation matches expected behavior
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.augmentation import get_transforms
from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from src.models.factory import create_model, list_models


# ============================================================
# End-to-end inference pipeline
# ============================================================

class TestInferencePipeline:
    """Test the full image → prediction flow."""

    @pytest.mark.parametrize("model_name", ["custom_cnn", "resnet_scratch", "mobilenet_scratch"])
    def test_image_to_prediction(self, model_name, dummy_image_np, sample_config):
        """
        Simulate what the API does:
        1. Load image as numpy array
        2. Apply test transforms
        3. Run model forward pass
        4. Get class prediction
        """
        # Build model
        model_cfg = sample_config["models"].get(model_name, {})
        model = create_model(model_name, model_cfg, num_classes=NUM_CLASSES)
        model.eval()

        # Preprocess (same as API)
        transform = get_transforms("test", image_size=224)
        tensor = transform(image=dummy_image_np)["image"].unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()

        # Validate output
        assert 0 <= pred_idx < NUM_CLASSES
        assert IDX_TO_CLASS[pred_idx] in CLASS_TO_IDX
        assert abs(probs.sum().item() - 1.0) < 1e-5  # Probabilities sum to 1

    def test_inference_deterministic(self, dummy_image_np, sample_config):
        """Same image + same model → same prediction (eval mode)."""
        model = create_model("custom_cnn", sample_config["models"]["custom_cnn"])
        model.eval()

        transform = get_transforms("test", image_size=224)
        tensor = transform(image=dummy_image_np)["image"].unsqueeze(0)

        with torch.no_grad():
            out1 = model(tensor)
            out2 = model(tensor)

        assert torch.allclose(out1, out2)


# ============================================================
# Checkpoint save/load round-trip
# ============================================================

class TestCheckpointRoundTrip:
    """Test that models can be saved and restored correctly."""

    @pytest.mark.parametrize("model_name", ["custom_cnn", "resnet_scratch"])
    def test_save_load_preserves_weights(self, model_name, sample_config, tmp_path):
        """Save a checkpoint, load it into a fresh model, compare outputs."""
        model_cfg = sample_config["models"].get(model_name, {})

        # Create and save model
        model_orig = create_model(model_name, model_cfg, num_classes=NUM_CLASSES)
        model_orig.eval()

        checkpoint_path = tmp_path / "model.pth"
        torch.save(
            {"model_state_dict": model_orig.state_dict(), "model_name": model_name},
            checkpoint_path,
        )

        # Load into fresh model
        model_loaded = create_model(model_name, model_cfg, num_classes=NUM_CLASSES)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_loaded.load_state_dict(checkpoint["model_state_dict"])
        model_loaded.eval()

        # Compare outputs
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_orig = model_orig(x)
            out_loaded = model_loaded(x)

        assert torch.allclose(out_orig, out_loaded, atol=1e-6)


# ============================================================
# Config-driven model creation
# ============================================================

class TestConfigDrivenCreation:
    """Verify config values actually affect the model."""

    def test_custom_cnn_blocks_from_config(self, sample_config):
        """Config num_conv_blocks should control architecture depth."""
        cfg_4 = {"num_conv_blocks": 4, "initial_filters": 32, "dropout": 0.5}
        cfg_3 = {"num_conv_blocks": 3, "initial_filters": 32, "dropout": 0.5}

        m4 = create_model("custom_cnn", cfg_4, num_classes=NUM_CLASSES)
        m3 = create_model("custom_cnn", cfg_3, num_classes=NUM_CLASSES)

        p4 = sum(p.numel() for p in m4.parameters())
        p3 = sum(p.numel() for p in m3.parameters())

        assert p4 > p3  # More blocks = more parameters

    def test_all_models_produce_num_classes_outputs(self, sample_config):
        """Every model should output exactly NUM_CLASSES logits."""
        x = torch.randn(1, 3, 224, 224)

        for model_name in list_models():
            model_cfg = sample_config["models"].get(model_name, {})
            model = create_model(model_name, model_cfg, num_classes=NUM_CLASSES)
            model.eval()

            with torch.no_grad():
                out = model(x)

            assert out.shape == (1, NUM_CLASSES), (
                f"{model_name} output shape {out.shape} != (1, {NUM_CLASSES})"
            )


# ============================================================
# Data pipeline integration
# ============================================================

class TestDataPipelineIntegration:
    """Test data pipeline → model integration."""

    def test_dataloader_feeds_model(self, dummy_split_dir, sample_config):
        """DataLoader output should be directly consumable by models."""
        from src.data.dataset import create_dataloaders

        loaders = create_dataloaders(
            dummy_split_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )

        model = create_model(
            "custom_cnn",
            sample_config["models"]["custom_cnn"],
            num_classes=NUM_CLASSES,
        )
        model.eval()

        images, labels = next(iter(loaders["train"]))

        with torch.no_grad():
            output = model(images)

        assert output.shape == (4, NUM_CLASSES)
        assert labels.shape == (4,)
        assert (labels >= 0).all() and (labels < NUM_CLASSES).all()
