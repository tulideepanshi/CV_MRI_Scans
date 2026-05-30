#!/usr/bin/env python3
"""
Training Loop — The engine that drives model learning.

WHAT THIS FILE DOES:
  Implements a complete training pipeline with:
    - Mixed precision training (AMP) for speed
    - Gradient clipping for stability
    - Cosine annealing learning rate schedule
    - Early stopping on validation F1 (not accuracy — clinical relevance)
    - Model checkpointing (save best + save last)
    - Per-epoch metrics logging
    - Optional mixup augmentation

WHY THESE CHOICES:
  Mixed precision (AMP):
    Forward pass in float16 (half the memory, 2x faster on modern GPUs).
    Backward pass and optimizer step in float32 (numerical stability).
    GradScaler handles the float16 → float32 gradient scaling automatically.

  Gradient clipping (max_norm=1.0):
    Caps gradient magnitude to prevent exploding gradients.
    Especially important for medical images where a single outlier
    (corrupted image, unusual anatomy) can produce a huge loss spike.

  Early stopping on val_f1_macro:
    Accuracy is misleading when classes are balanced but importance differs.
    A model that's 95% accurate but misses 30% of gliomas is dangerous.
    F1 macro averages F1 across all classes equally, so every class matters.

  Cosine annealing:
    LR starts high → decays smoothly to near-zero following a cosine curve.
    No sudden drops (like step decay), so training is smoother.

Usage:
    from src.training.trainer import Trainer

    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train()
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score

from src.data.augmentation import mixup_data


class EarlyStopping:
    """
    Stop training when validation metric stops improving.

    How it works:
      - Track the best validation metric seen so far
      - If the metric hasn't improved for `patience` epochs, stop
      - "Improved" means exceeded best + min_delta (avoids noise triggers)

    Why patience=7:
      Too low (1-2): stops on normal fluctuations, misses later improvements.
      Too high (15+): wastes compute training a converged model.
      7 is a common sweet spot — enough to see through noise, short enough
      to catch genuine plateaus.
    """

    def __init__(
        self,
        patience: int = 7,
        mode: str = "max",
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Complete training pipeline for brain tumor classification.

    Handles: training loop, validation, checkpointing, early stopping,
    mixed precision, gradient clipping, learning rate scheduling, logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device = None,
        output_dir: str = "models",
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        class_weights: torch.Tensor = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 30)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.grad_clip = train_cfg.get("gradient_clipping", 1.0)
        self.use_amp = train_cfg.get("mixed_precision", True) and self.device.type == "cuda"

        # Loss function with optional class weighting
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer — AdamW (Adam with decoupled weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler — Cosine Annealing
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=sched_cfg.get("T_max", self.num_epochs),
            eta_min=sched_cfg.get("eta_min", 1e-5),
        )

        # Mixed precision scaler (device_type required in new API)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = None
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 7),
                mode=es_cfg.get("mode", "max"),
            )

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "lr": [],
            "epoch_time": [],
        }

    def train_one_epoch(self) -> dict:
        """
        Train for one epoch. Returns dict with loss and accuracy.

        One epoch = one pass through the entire training dataset.

        For each batch:
          1. Forward pass (in float16 if AMP enabled)
          2. Compute loss (with optional mixup)
          3. Backward pass (scaled gradients if AMP)
          4. Clip gradients (prevent explosion)
          5. Optimizer step (update weights)
          6. Scheduler step happens after all batches (in train())
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Optional mixup
            if self.use_mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.mixup_alpha
                )

            # Forward pass with automatic mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)

                if self.use_mixup:
                    # Mixup loss: weighted combination of two class losses
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for AMP)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            if not self.use_mixup:
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            else:
                # For mixup, accuracy is approximate (use dominant label)
                total += labels_a.size(0)
                correct += (lam * predicted.eq(labels_a).sum().item() +
                            (1 - lam) * predicted.eq(labels_b).sum().item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Validate on the validation set. Returns loss, accuracy, F1 macro.

        @torch.no_grad(): disables gradient computation.
        Why: validation doesn't need gradients (no weight updates).
        Saves memory and speeds up computation.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        total = len(all_labels)
        epoch_loss = running_loss / total
        epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "f1_macro": epoch_f1,
        }

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: dict):
        """Save model checkpoint with full training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "history": self.history,
            "config": self.config,
        }
        torch.save(checkpoint, self.output_dir / filename)

    def train(self) -> dict:
        """
        Full training loop with validation, early stopping, and checkpointing.

        Returns training history dict.
        """
        print(f"\nTraining on {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Epochs: {self.num_epochs}, LR: {self.lr}, Batch size: {self.config.get('training', {}).get('batch_size', 32)}")
        print(f"Mixup: {self.use_mixup}" + (f" (alpha={self.mixup_alpha})" if self.use_mixup else ""))
        print("=" * 60)

        best_f1 = 0.0

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch()

            # Validate
            val_metrics = self.validate()

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_f1_macro"].append(val_metrics["f1_macro"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # Print epoch summary
            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"F1: {val_metrics['f1_macro']:.4f} | "
                f"LR: {current_lr:.6f} | {epoch_time:.1f}s"
            )

            # Save best model
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                self.save_checkpoint("best_model.pth", epoch, val_metrics)
                print(f"  → New best F1: {best_f1:.4f} (saved)")

            # Early stopping check
            if self.early_stopping:
                if self.early_stopping(val_metrics["f1_macro"]):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best val F1: {best_f1:.4f}")
                    break

        # Save final model
        self.save_checkpoint("last_model.pth", epoch, val_metrics)

        # Save training history as JSON
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
        print(f"History saved to {history_path}")

        return self.history
