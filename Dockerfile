##############################################################################
# Dockerfile — Brain Tumor Classification API
#
# Multi-stage build for a lean production image:
#   Stage 1 (builder): install all Python dependencies in a virtual env
#   Stage 2 (runtime): copy only the venv + app code — no compilers, no pip
#
# WHY MULTI-STAGE:
#   Single stage with pip install → ~2.5 GB image (compilers, caches, headers)
#   Multi-stage → ~1.2 GB (just the runtime + model weights)
#   Smaller image = faster deploys, less attack surface, cheaper storage.
#
# WHY python:3.11-slim (not alpine):
#   Alpine uses musl libc. PyTorch and numpy expect glibc. Building them
#   on Alpine requires hours of compilation and often fails. Slim is the
#   right choice for ML workloads — small enough, compatible with everything.
#
# Usage:
#   docker build -t brain-tumor-api .
#   docker run -p 8000:8000 -v ./models:/app/models brain-tumor-api
##############################################################################

# ========================
# Stage 1: Builder
# ========================
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install system dependencies needed for compilation
# (These won't be in the final image — that's the point of multi-stage)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# Copy requirements first for layer caching — if requirements don't change,
# Docker reuses this layer even if source code changes (saves rebuild time)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ========================
# Stage 2: Runtime
# ========================
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="Paras Gulati <parasgulati8@gmail.com>" \
      description="Brain Tumor MRI Classification API" \
      version="1.0.0"

# Security: run as non-root user
# Why: if the container is compromised, the attacker doesn't have root.
# This is a Docker security best practice and often required for compliance.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal runtime dependencies
# libgomp1: OpenMP for PyTorch CPU parallelism
# libgl1: OpenCV dependency (used by albumentations)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code (only what's needed — .dockerignore handles the rest)
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create directories for runtime
RUN mkdir -p logs models data/splits && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables with sensible defaults
# These can be overridden in docker-compose.yml or at runtime
ENV MODEL_PATH="models/best_model.pth" \
    MODEL_NAME="efficientnet_pretrained" \
    CONFIG_PATH="configs/config.yaml" \
    IMAGE_SIZE="224" \
    MAX_FILE_SIZE_MB="10" \
    LOG_PREDICTIONS="true" \
    PYTHONPATH="/app"

# Expose API port
EXPOSE 8000

# Health check — Docker uses this to determine if the container is healthy
# Checks every 30s, times out after 10s, retries 3 times before marking unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
# Workers=1 for CPU inference (model isn't thread-safe for training, but
# inference with torch.no_grad() is fine with a single worker).
# For GPU: still use 1 worker (GPU is the bottleneck, not the web server).
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
