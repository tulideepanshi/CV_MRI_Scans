"""
Tests for the FastAPI inference server.

Uses FastAPI's TestClient (synchronous) to test endpoints
without starting a real server. Model is mocked to avoid
needing a checkpoint file.

Requires: pip install httpx  (TestClient dependency)

Covers:
  - Health check endpoint
  - Model info endpoint
  - Predict endpoint (valid and invalid inputs)
  - File validation (size, type)
"""

import io
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

httpx = pytest.importorskip("httpx", reason="httpx required for API tests")

from src.data.augmentation import get_transforms
from src.data.dataset import NUM_CLASSES


# ============================================================
# No-op lifespan to replace the real one (which loads a model)
# ============================================================

@asynccontextmanager
async def _noop_lifespan(app):
    """Skip model loading during tests — state is injected by fixtures."""
    yield


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_model():
    """Create a mock model that returns fixed logits."""
    model = MagicMock()
    # Return logits where class 0 (glioma) has highest score
    logits = torch.tensor([[2.0, 0.5, 0.1, 0.3]])
    model.return_value = logits
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def client(mock_model):
    """
    Create a FastAPI TestClient with mocked model state.

    Injects a mock model into the app's global state and replaces the
    lifespan with a no-op so we don't need a real checkpoint file.
    """
    from fastapi.testclient import TestClient
    from src.api import app as app_module

    # Inject mock state directly into the module-level singleton
    app_module.state.model = mock_model
    app_module.state.device = torch.device("cpu")
    app_module.state.transform = get_transforms("test", image_size=224)
    app_module.state.model_name = "test_model"
    app_module.state.checkpoint_path = "models/test.pth"

    # Replace lifespan with no-op
    app_module.app.router.lifespan_context = _noop_lifespan

    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def dummy_image_bytes():
    """Create a valid JPEG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# Health endpoint
# ============================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Healthy state should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_no_model(self):
        """Without model, should return 503."""
        from fastapi.testclient import TestClient
        from src.api import app as app_module

        # Set state to no-model
        app_module.state.model = None
        app_module.state.device = torch.device("cpu")
        app_module.state.model_name = ""

        app_module.app.router.lifespan_context = _noop_lifespan

        with TestClient(app_module.app) as c:
            response = c.get("/health")
            assert response.status_code == 503


# ============================================================
# Model info endpoint
# ============================================================

class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info(self, client):
        """Should return model metadata."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == NUM_CLASSES
        assert len(data["class_names"]) == NUM_CLASSES
        assert "glioma" in data["class_names"]


# ============================================================
# Predict endpoint
# ============================================================

class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_image(self, client, dummy_image_bytes):
        """Valid JPEG upload should return a prediction."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_ms" in data
        assert data["prediction"] in ["glioma", "meningioma", "notumor", "pituitary"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == NUM_CLASSES

    def test_predict_png_accepted(self, client):
        """PNG images should also be accepted."""
        img = Image.fromarray(
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200

    def test_predict_invalid_extension(self, client):
        """Non-image file extension should return 400."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_no_file(self, client):
        """Missing file should return 422 (validation error)."""
        response = client.post("/predict")
        assert response.status_code == 422


# ============================================================
# File validation
# ============================================================

class TestFileValidation:
    """Tests for upload validation logic."""

    def test_validate_extension(self):
        """validate_upload should reject unsupported extensions."""
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from src.api.app import validate_upload

        file = MagicMock()
        file.filename = "test.bmp"
        file.content_type = "image/bmp"

        with pytest.raises(HTTPException) as exc_info:
            validate_upload(file)
        assert exc_info.value.status_code == 400

    def test_validate_content_type(self):
        """validate_upload should reject non-image content types."""
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from src.api.app import validate_upload

        file = MagicMock()
        file.filename = "test.jpg"
        file.content_type = "application/pdf"

        with pytest.raises(HTTPException) as exc_info:
            validate_upload(file)
        assert exc_info.value.status_code == 400
