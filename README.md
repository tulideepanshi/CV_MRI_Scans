# Brain Tumor MRI Classification

End-to-end deep learning pipeline for detecting and classifying brain tumors from MRI scans. Built with PyTorch, featuring 5 model architectures, clinical-grade evaluation metrics, Grad-CAM interpretability, and a Dockerized deployment stack.
Quick look at the results at https://tulideepanshi.github.io/CV_MRI_Scans/

## Problem Statement

Classify brain MRI images into 4 categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. The system provides both predictions and visual explanations (Grad-CAM heatmaps) showing where the model focuses — critical for clinical trust and decision support.

## Project Structure

```
brain-tumor-classification/
├── configs/
│   └── config.yaml              # Master configuration (data, models, training, deployment)
├── data/
│   ├── splits/                  # Train/val/test splits (70/15/15)
│   └── duplicates.json          # Near-duplicate pair mappings
├── datasets/                    # Raw Kaggle data (Training/ & Testing/)
├── deploy/
│   ├── nginx/nginx.conf         # Reverse proxy + frontend serving + rate limiting
│   └── prometheus/              # Monitoring configuration
├── frontend/
│   └── index.html               # Clinical-grade web UI (drag-and-drop MRI analysis)
├── docs/
│   ├── eda/                     # Exploratory data analysis plots
│   ├── eda_post_split/          # Post-split distribution analysis
│   └── eval/                    # Evaluation reports and visualizations
├── models/                      # Trained model checkpoints
├── scripts/
│   ├── train.py                 # Training CLI (single model or all 5)
│   └── evaluate.py              # Evaluation CLI (metrics + Grad-CAM)
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI inference server
│   ├── data/
│   │   ├── augmentation.py      # Conservative vs aggressive augmentation pipelines
│   │   ├── compare_splits.py    # Kaggle vs cluster-aware split comparison
│   │   ├── dataset.py           # PyTorch Dataset + DataLoader factory
│   │   ├── dedup.py             # Near-duplicate detection (pHash + embeddings)
│   │   ├── download.py          # Kaggle dataset downloader
│   │   ├── explore.py           # EDA with raw and post-split modes
│   │   └── split.py             # Cluster-aware stratified splitting
│   ├── evaluation/
│   │   └── evaluator.py         # Clinical metrics suite + Grad-CAM
│   ├── models/
│   │   ├── custom_cnn.py        # 4-block CNN baseline (~390K params)
│   │   ├── resnet_scratch.py    # ResNet-18 from scratch (~11.2M params)
│   │   ├── mobilenet_scratch.py # MobileNet-V1 style (~3.2M params)
│   │   ├── pretrained.py        # PretrainedResNet + PretrainedEfficientNet
│   │   └── factory.py           # Model registry and creation factory
│   ├── security/
│   │   └── privacy.py           # DICOM anonymization, AES-256 encryption, audit logging
│   ├── training/
│   │   └── trainer.py           # Training loop (AMP, gradient clipping, early stopping)
│   └── utils/
│       └── reproducibility.py   # Seed management, device detection
├── docker-compose.yml           # Multi-service deployment stack
├── Dockerfile                   # Multi-stage build for production API
└── requirements.txt             # Python dependencies
```

## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

```bash
python -m src.data.download
```

### 3. Data Pipeline

```bash
# Exploratory data analysis
python -m src.data.explore --mode raw

# Near-duplicate detection
python -m src.data.dedup

# Cluster-aware stratified split
python -m src.data.split

# Post-split EDA
python -m src.data.explore --mode split
```

### 4. Train Models

```bash
# Train all 5 models
python scripts/train.py --model all --augmentation conservative

# Train a specific model
python scripts/train.py --model efficientnet_pretrained --augmentation conservative

# Quick test run
python scripts/train.py --model custom_cnn --epochs 3
```

### 5. Evaluate

```bash
# Evaluate all trained models
python scripts/evaluate.py --model all

# Single model with Grad-CAM
python scripts/evaluate.py --model efficientnet_pretrained
```

### 6. Deploy

```bash
# Copy best model checkpoint
cp models/efficientnet_pretrained_conservative/best_model.pth models/best_model.pth

# Build and start the full stack (API + frontend + nginx)
docker compose up -d --build

# Open the clinical frontend
open http://localhost        # macOS
# or visit http://localhost in your browser

# Direct API access (bypassing nginx)
curl -X POST http://localhost:8000/predict \
     -F "file=@test_image.jpg"

# API with Grad-CAM heatmap
curl -X POST http://localhost:8000/predict/gradcam \
     -F "file=@test_image.jpg"

# View API docs (Swagger)
open http://localhost/docs

# Stop everything
docker compose down
```

## Model Architectures

| Model | Params | Type | Key Feature |
|-------|--------|------|-------------|
| Custom CNN | ~390K | From scratch | 4-block baseline with GAP |
| ResNet-18 Scratch | ~11.2M | From scratch | Residual connections, skip connections |
| MobileNet Scratch | ~3.2M | From scratch | Depthwise separable convolutions |
| ResNet-18 Pretrained | ~11.7M | Transfer learning | ImageNet weights, full fine-tuning |
| EfficientNet-B0 Pretrained | ~5.3M | Transfer learning | Squeeze-and-excitation, compound scaling |

## Data Pipeline Decisions

**Near-Duplicate Detection**: Perceptual hashing (pHash) identifies near-duplicate images that could cause data leakage between train/test splits. Uses Union-Find to build duplicate clusters.

**Cluster-Aware Splitting**: All images in a duplicate cluster go to the same split (train, val, or test). Weighted greedy bin-packing ensures balanced class distribution across splits despite cluster-level assignment.

**Augmentation Ablation**: Two augmentation strategies are compared:
- **Conservative**: rotation, scale, brightness/contrast, elastic transform — all clinically safe (no flips, preserves laterality)
- **Aggressive**: adds horizontal flip, cutout, CLAHE, Gaussian blur — stronger regularization but sacrifices clinical faithfulness

## Training Configuration

- Optimizer: AdamW (weight decay 0.0001)
- Scheduler: Cosine annealing (LR 0.001 → 0.00001)
- Early stopping: patience 7, monitors val F1-macro
- Mixed precision: AMP with GradScaler (CUDA) / disabled on MPS
- Gradient clipping: max norm 1.0
- Optional: Mixup augmentation (alpha=0.2)

## Evaluation Metrics

Clinical metrics beyond standard accuracy:
- **Sensitivity (recall)**: fraction of actual tumors correctly detected
- **Specificity**: fraction of healthy patients correctly cleared
- **F1-macro**: balanced metric across all classes
- **ROC-AUC**: one-vs-rest, threshold-independent performance
- **Grad-CAM**: heatmap visualization of model attention regions

## Security & Privacy

- **DICOM Anonymization**: removes/hashes PHI fields before ML processing
- **AES-256-GCM Encryption**: data at rest protection with authenticated encryption
- **Audit Logging**: JSON Lines append-only log of all data access and predictions
- **API Security**: file size limits, extension whitelist, rate limiting (nginx), non-root container

## Docker Stack

- **api**: FastAPI inference server (uvicorn, non-root user)
- **nginx**: Reverse proxy + clinical frontend serving + rate limiting (10 req/s per IP)
- **monitor**: Prometheus metrics (optional, `--profile monitoring`)
- **frontend**: Drag-and-drop MRI analysis UI with Grad-CAM visualization (served by nginx)

Multi-stage Dockerfile produces a ~1.2GB image (vs ~2.5GB single-stage).

Architecture: `Browser → nginx:80 → frontend (static HTML) or → api:8000 (inference)`

## Tech Stack

- **Framework**: PyTorch 2.x, torchvision
- **Augmentation**: albumentations
- **API**: FastAPI, uvicorn
- **Deployment**: Docker Compose, nginx
- **Security**: cryptography (AES-256-GCM), pydicom
- **Evaluation**: scikit-learn, matplotlib
- **Data**: pandas, numpy, Pillow

## Author

**Deepanshi Tuli** — tulideepanshi@gmail.com
