"""
PyTorch MLP classifier with GPU acceleration (RTX 3050 / CUDA 12.4).

Uses focal loss when gamma > 0 to handle class imbalance at the loss level.
Supports early stopping on validation loss.
Falls back to CPU automatically if CUDA is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 reduces to standard cross-entropy.
    alpha handles class imbalance weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for positive class

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary output logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class MLPClassifier_(BaseClassifier):
    """
    GPU-accelerated PyTorch MLP with focal loss support.

    GPU acceleration: automatically uses CUDA if available (RTX 3050).
    Falls back to CPU transparently.

    Parameters
    ----------
    hidden_sizes  : list of hidden layer widths
    dropout       : dropout rate (applied after each hidden layer)
    lr            : Adam learning rate
    max_epochs    : maximum training epochs
    batch_size    : mini-batch size (256 fits comfortably on 4GB VRAM)
    patience      : early stopping patience (epochs without val loss improvement)
    gamma         : focal loss gamma (0 = standard cross-entropy)
    device        : 'cuda' | 'cpu' | None (auto-detect)
    seed          : random seed
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        max_epochs: int = 200,
        batch_size: int = 256,
        patience: int = 15,
        gamma: float = 0.0,
        device: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self._hidden_sizes = hidden_sizes or [64, 32]
        self._dropout = dropout
        self._lr = lr
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._patience = patience
        self._gamma = gamma
        self._seed = seed

        # Device selection — prefer CUDA for RTX 3050
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info("MLPClassifier using device: %s", self._device)

        self._model: Optional[_MLP] = None
        self._classes_: Optional[np.ndarray] = None
        self._epoch_val_losses: list[float] = []
        self._epoch_train_losses: list[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        self._classes_ = np.unique(y)
        input_dim = X.shape[1]

        self._model = _MLP(input_dim, self._hidden_sizes, self._dropout).to(self._device)

        # Focal loss — alpha from class imbalance if sample_weight provided
        alpha = None
        if sample_weight is not None:
            pos_weight = float(sample_weight[y == 1].mean())
            neg_weight = float(sample_weight[y == 0].mean())
            alpha = pos_weight / (pos_weight + neg_weight)
        criterion = FocalLoss(gamma=self._gamma, alpha=alpha)

        optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )

        # Build DataLoader
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32)
            dataset = TensorDataset(X_t, y_t, w_t)
        else:
            dataset = TensorDataset(X_t, y_t)

        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True,
            pin_memory=(self._device.type == "cuda"),
            num_workers=0,
        )

        # Validation tensors (for early stopping)
        X_val_t = y_val_t = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self._device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self._device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(
            "Training MLP: %d samples, %d features, device=%s, epochs=%d",
            X.shape[0], input_dim, self._device, self._max_epochs,
        )

        for epoch in range(self._max_epochs):
            self._model.train()
            epoch_loss = 0.0
            for batch in loader:
                if len(batch) == 3:
                    xb, yb, wb = batch
                    xb, yb, wb = xb.to(self._device), yb.to(self._device), wb.to(self._device)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    wb = None

                optimizer.zero_grad()
                logits = self._model(xb)

                if wb is not None:
                    # Apply focal loss with sample weights:
                    # compute focal loss per-sample, then weight by wb
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logits, yb, reduction="none"
                    )
                    if self._gamma > 0:
                        probs = torch.sigmoid(logits)
                        p_t = probs * yb + (1 - probs) * (1 - yb)
                        focal_weight = (1 - p_t) ** self._gamma
                        bce = focal_weight * bce
                    loss = (bce * wb).mean()
                else:
                    loss = criterion(logits, yb)

                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)
            self._epoch_train_losses.append(avg_train_loss)

            # Validation loss for early stopping
            if X_val_t is not None:
                self._model.eval()
                with torch.no_grad():
                    val_logits = self._model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                self._epoch_val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self._patience:
                        logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch, val_loss)
                        break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        logger.info(
            "MLP training complete. Final train_loss=%.4f, best_val_loss=%.4f",
            self._epoch_train_losses[-1] if self._epoch_train_losses else float("nan"),
            best_val_loss,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model(X_t)
            probs_pos = torch.sigmoid(logits).cpu().numpy()
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def get_params(self) -> dict:
        return {
            "model": "mlp",
            "hidden_sizes": self._hidden_sizes,
            "dropout": self._dropout,
            "lr": self._lr,
            "max_epochs": self._max_epochs,
            "batch_size": self._batch_size,
            "patience": self._patience,
            "gamma": self._gamma,
            "device": str(self._device),
            "seed": self._seed,
        }

    @property
    def epoch_val_losses(self) -> list[float]:
        return self._epoch_val_losses

    @property
    def epoch_train_losses(self) -> list[float]:
        return self._epoch_train_losses
