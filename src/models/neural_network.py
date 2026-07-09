"""
src/models/neural_network.py
Three-layer feedforward neural network in PyTorch.
Matches the architecture described in the project report.
Binary primary, multiclass secondary.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# ── Architecture ──────────────────────────────────────────────────────────

class FeedForwardIDS(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list,
                 output_dim: int, dropout: float = config.NN_DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────

class NeuralNetworkModel:
    """Wrapper to keep the same train/predict interface as sklearn models."""

    def __init__(self, input_dim: int, output_dim: int = 2,
                 binary: bool = True):
        self.binary     = binary
        self.output_dim = output_dim
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[neural_network] Using device: {self.device}")

        self.model = FeedForwardIDS(
            input_dim=input_dim,
            hidden_layers=config.NN_HIDDEN_LAYERS,
            output_dim=1 if binary else output_dim,
            dropout=config.NN_DROPOUT,
        ).to(self.device)

        self.trained = False

    def _to_tensors(self, X, y=None):
        X_t = torch.FloatTensor(X).to(self.device)
        if y is not None:
            if self.binary:
                y_t = torch.FloatTensor(y).unsqueeze(1).to(self.device)
            else:
                y_t = torch.LongTensor(y).to(self.device)
            return X_t, y_t
        return X_t

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):

        X_t, y_t = self._to_tensors(X_train, y_train)
        dataset   = TensorDataset(X_t, y_t)
        loader    = DataLoader(dataset, batch_size=config.NN_BATCH_SIZE, shuffle=True)

        if self.binary:
            # Weighted BCE for class imbalance
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                                      dtype=torch.float).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = Adam(self.model.parameters(), lr=config.NN_LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=False)

        best_val_loss = float("inf")
        patience_ctr  = 0
        best_state    = None

        for epoch in range(1, config.NN_EPOCHS + 1):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out  = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._val_loss(X_val, y_val, criterion)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_ctr  = 0
                else:
                    patience_ctr += 1

                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:>3}/{config.NN_EPOCHS}  "
                          f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

                if patience_ctr >= config.NN_PATIENCE:
                    print(f"[neural_network] Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:>3}/{config.NN_EPOCHS}  loss={avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.trained = True
        print("[neural_network] Training complete.")
        return self

    def _val_loss(self, X_val, y_val, criterion):
        self.model.eval()
        with torch.no_grad():
            X_t, y_t = self._to_tensors(X_val, y_val)
            out  = self.model(X_t)
            loss = criterion(out, y_t)
        return loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensors(X)
            out = self.model(X_t)
            if self.binary:
                preds = (torch.sigmoid(out) >= 0.5).squeeze(1).long()
            else:
                preds = torch.argmax(out, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensors(X)
            out = self.model(X_t)
            if self.binary:
                prob_pos = torch.sigmoid(out).squeeze(1)
                prob_neg = 1 - prob_pos
                probs = torch.stack([prob_neg, prob_pos], dim=1)
            else:
                probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()

    def save(self, path: str = None):
        path = path or os.path.join(config.MODEL_DIR, "neural_network.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "binary":      self.binary,
            "output_dim":  self.output_dim,
            "input_dim":   self.model.net[0].in_features,
        }, path)
        print(f"[neural_network] Model saved to {path}")

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(config.MODEL_DIR, "neural_network.pt")
        ckpt = torch.load(path, map_location="cpu")
        obj  = cls(input_dim=ckpt["input_dim"],
                   output_dim=ckpt["output_dim"],
                   binary=ckpt["binary"])
        obj.model.load_state_dict(ckpt["model_state"])
        obj.trained = True
        return obj
