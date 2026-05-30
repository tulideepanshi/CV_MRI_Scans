#!/usr/bin/env python3
"""
FastAPI Inference Server — Production-ready brain tumor classification API.

WHAT THIS FILE DOES:
  Serves a trained PyTorch model as a REST API with:
    - POST /predict: Upload an MRI image, get classification + confidence
    - POST /predict/gradcam: Same + Grad-CAM heatmap overlay
    - GET /health: Health check (is the model loaded? is GPU available?)
    - GET /model/info: Model metadata (architecture, classes, metrics)

ARCHITECTURE DECISIONS:
  FastAPI (not Flask):
    - Async by default → handles concurrent requests without blocking
    - Automatic OpenAPI/Swagger docs at /docs
    - Pydantic validation → type-safe request/response schemas
    - Built-in file upload handling with UploadFile

  Model loading strategy:
    - Load ONCE at startup (app lifespan context manager)
    - Keep in memory — no disk I/O per request
    - Thread-safe for inference (model.eval() + torch.no_grad())

  Image preprocessing:
    - Same pipeline as test-time (resize → normalize → tensor)
    - Validates file type and size BEFORE processing
    - Handles RGB conversion (grayscale MRI → 3-channel)

  Response format:
    - prediction: string class name
    - confidence: float [0, 1]
    - probabilities: dict {class_name: probability} for all classes
    - processing_time_ms: float (useful for SLA monitoring)

SECURITY:
  - File size limit (10MB default) — prevents memory exhaustion
  - Extension whitelist — only .jpg, .jpeg, .png, .dcm
  - Input validation via Pydantic
  - No user data stored — stateless inference
  - Audit logging for every prediction (configurable)

Usage:
    # Start server
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

    # Or with Docker
    docker compose up api
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Project imports — these work when PYTHONPATH includes project root
from src.data.augmentation import get_transforms
from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from src.models.factory import create_model
from src.evaluation.evaluator import GradCAM

# ============================================================
# Configuration
# ============================================================

# Defaults — overridden by environment variables in Docker
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
MODEL_NAME = os.getenv("MODEL_NAME", "efficientnet_pretrained")
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", mode="a") if Path("logs").exists()
        else logging.StreamHandler(),
    ],
)
logger = logging.getLogger("brain-tumor-api")


# ============================================================
# Response schemas
# ============================================================

class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    prediction: str               # Class name: "glioma", "meningioma", etc.
    confidence: float             # Probability of predicted class [0, 1]
    probabilities: dict[str, float]  # All class probabilities
    processing_time_ms: float     # Inference time in milliseconds

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "glioma",
                "confidence": 0.94,
                "probabilities": {
                    "glioma": 0.94,
                    "meningioma": 0.03,
                    "notumor": 0.02,
                    "pituitary": 0.01,
                },
                "processing_time_ms": 45.2,
            }
        }


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    model_loaded: bool
    device: str
    model_name: str


class ModelInfoResponse(BaseModel):
    """Response schema for /model/info endpoint."""
    model_name: str
    num_classes: int
    class_names: list[str]
    image_size: int
    device: str
    checkpoint_path: str


# ============================================================
# Global state (populated at startup)
# ============================================================

class ModelState:
    """Container for loaded model and preprocessing pipeline."""
    model: torch.nn.Module = None
    device: torch.device = None
    transform = None
    model_name: str = ""
    checkpoint_path: str = ""


state = ModelState()


# ============================================================
# Model loading
# ============================================================

def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load trained model from checkpoint file.

    Same as evaluate.py but without config dependency — reads model
    architecture from MODEL_NAME env var.
    """
    import yaml

    # Try to load config for model-specific params
    config = {}
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("models", {}).get(model_name, {})
    num_classes = config.get("data", {}).get("num_classes", NUM_CLASSES)

    model = create_model(model_name, model_cfg, num_classes)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model_name} from {checkpoint_path}")
    logger.info(f"Device: {device}")

    return model


# ============================================================
# App lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model at startup, clean up at shutdown.

    Why lifespan (not @app.on_event):
      FastAPI deprecated on_event in favor of lifespan context managers.
      The yield separates startup (before yield) from shutdown (after yield).
    """
    # --- Startup ---
    logger.info("Starting Brain Tumor Classification API...")

    # Detect device
    if torch.cuda.is_available():
        state.device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        state.device = torch.device("mps")
    else:
        state.device = torch.device("cpu")

    # Load model
    checkpoint_path = MODEL_PATH
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Set MODEL_PATH environment variable to a valid .pth file")
        # Don't crash — allow health check to report the error
    else:
        state.model = load_model_from_checkpoint(MODEL_NAME, checkpoint_path, state.device)
        state.model_name = MODEL_NAME
        state.checkpoint_path = checkpoint_path

    # Build preprocessing pipeline (same as test-time)
    state.transform = get_transforms("test", image_size=IMAGE_SIZE)

    logger.info("API ready.")

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    yield

    # --- Shutdown ---
    logger.info("Shutting down API...")
    state.model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Brain Tumor MRI Classification API",
    description=(
        "Upload a brain MRI image to get tumor classification. "
        "Supports glioma, meningioma, pituitary tumor, and no-tumor detection. "
        "Includes Grad-CAM heatmap visualization for model interpretability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper functions
# ============================================================

def validate_upload(file: UploadFile) -> None:
    """Validate uploaded file before processing."""
    # Check extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'. Must be an image.",
        )


async def preprocess_image(file: UploadFile) -> torch.Tensor:
    """
    Read uploaded image and apply test-time preprocessing.

    Pipeline: bytes → PIL.Image → RGB → numpy → albumentations → tensor
    """
    # Read file bytes
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f} MB). "
                   f"Max: {MAX_FILE_SIZE_MB} MB.",
        )

    # Open and convert to RGB
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not open image: {str(e)}",
        )

    # Apply transforms
    img_np = np.array(img)
    transformed = state.transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0)  # Add batch dimension

    return tensor


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check — used by Docker HEALTHCHECK and load balancers.

    Returns 200 if model is loaded, 503 if not.
    """
    is_loaded = state.model is not None
    response = HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        device=str(state.device) if state.device else "none",
        model_name=state.model_name,
    )
    if not is_loaded:
        return JSONResponse(content=response.model_dump(), status_code=503)
    return response


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=state.model_name,
        num_classes=NUM_CLASSES,
        class_names=list(CLASS_TO_IDX.keys()),
        image_size=IMAGE_SIZE,
        device=str(state.device),
        checkpoint_path=state.checkpoint_path,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify a brain MRI image.

    Upload a JPEG or PNG image. Returns the predicted tumor class
    and confidence scores for all classes.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate
    validate_upload(file)

    # Preprocess
    start_time = time.time()
    tensor = await preprocess_image(file)
    tensor = tensor.to(state.device)

    # Inference
    with torch.no_grad():
        output = state.model(tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    processing_time = (time.time() - start_time) * 1000  # ms

    # Build response
    probabilities = {
        IDX_TO_CLASS[i]: round(probs[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    prediction = IDX_TO_CLASS[pred_idx]

    # Audit log
    if LOG_PREDICTIONS:
        logger.info(
            f"PREDICTION | file={file.filename} | "
            f"result={prediction} | confidence={confidence:.4f} | "
            f"time={processing_time:.1f}ms"
        )

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        probabilities=probabilities,
        processing_time_ms=round(processing_time, 2),
    )


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Classify + return Grad-CAM heatmap overlay.

    Returns the same prediction JSON plus a base64-encoded PNG of the
    Grad-CAM overlay, showing where the model focused.

    Why base64 (not a separate image endpoint):
      Single request = single response. The frontend doesn't need to
      make two calls (one for prediction, one for heatmap). The base64
      string can be directly used as an <img src="data:image/png;base64,...">.
    """
    import base64

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    validate_upload(file)

    start_time = time.time()

    # Read image for both inference and display
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    # Preprocess for model
    transformed = state.transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(state.device)
    tensor.requires_grad_(True)

    # Get target layer for Grad-CAM
    if hasattr(state.model, "get_last_conv_layer"):
        target_layer = state.model.get_last_conv_layer()
    else:
        raise HTTPException(
            status_code=500,
            detail="Model doesn't support Grad-CAM (no get_last_conv_layer method)",
        )

    # Initialize Grad-CAM
    grad_cam = GradCAM(state.model, target_layer)

    # Forward + backward for Grad-CAM
    with torch.enable_grad():
        output = state.model(tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        heatmap = grad_cam.generate(tensor, target_class=pred_idx)

    grad_cam.remove_hooks()

    # Create overlay image
    # Resize heatmap to original image dimensions
    h, w = img_np.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply jet colormap
    import matplotlib.pyplot as plt
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]

    # Alpha blend
    alpha = 0.4
    overlay = (1 - alpha) * (img_np / 255.0) + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # Encode as base64 PNG
    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    processing_time = (time.time() - start_time) * 1000

    probabilities = {
        IDX_TO_CLASS[i]: round(probs[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    prediction = IDX_TO_CLASS[pred_idx]

    if LOG_PREDICTIONS:
        logger.info(
            f"GRADCAM | file={file.filename} | "
            f"result={prediction} | confidence={confidence:.4f} | "
            f"time={processing_time:.1f}ms"
        )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "processing_time_ms": round(processing_time, 2),
        "gradcam_overlay_base64": heatmap_b64,
    }
