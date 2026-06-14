from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import compute_metrics


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    n_seq = n_samples - seq_len + 1
    X_seq = np.empty((n_seq, seq_len, X.shape[1]), dtype=X.dtype)
    y_seq = np.empty(n_seq, dtype=y.dtype)
    for i in range(n_seq):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len - 1]
    return X_seq, y_seq


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int = 10,
    hidden_size: int = 32,
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 64,
    val_split: float = 0.2,
) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

    n_val = max(1, int(len(X_train_seq) * val_split))
    X_val_seq, y_val_seq = X_train_seq[-n_val:], y_train_seq[-n_val:]
    X_train_seq, y_train_seq = X_train_seq[:-n_val], y_train_seq[:-n_val]

    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test_seq, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=X_train.shape[1], hidden_size=hidden_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for _epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
                n_batches += 1
        val_loss /= n_batches

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    model.eval()
    y_prob_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            y_prob_list.append(torch.sigmoid(output).cpu().numpy())
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = compute_metrics(y_test_seq, y_pred, y_prob)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }
