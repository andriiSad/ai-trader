from __future__ import annotations

import logging
import multiprocessing
import pickle
import tempfile

import numpy as np

from src.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _lstm_worker(payload: bytes, result_path: str) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    p = pickle.loads(payload)
    X_train, y_train, X_test, y_test = p["X_train"], p["y_train"], p["X_test"], p["y_test"]
    seq_len, hidden_size, max_epochs, patience, batch_size, val_split = (
        p["seq_len"],
        p["hidden_size"],
        p["max_epochs"],
        p["patience"],
        p["batch_size"],
        p["val_split"],
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train_seq, y_train_seq = _create_sequences_np(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = _create_sequences_np(X_test, y_test, seq_len)

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

    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, hs: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hs, 1, batch_first=True)
            self.fc = nn.Linear(hs, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMModel(input_size=X_train.shape[1], hs=hidden_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
                n_val_batches += 1
        val_loss /= n_val_batches

        if epoch == 0 or epoch % 10 == 0 or val_loss < best_val_loss:
            print(
                f"    Epoch {epoch:3d}: train_loss={train_loss / n_batches:.4f}, val_loss={val_loss:.4f}, patience={patience_counter}/{patience}",
                flush=True,
            )

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
            output = model(X_batch.to(device))
            y_prob_list.append(torch.sigmoid(output).cpu().numpy())
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = compute_metrics(y_test_seq, y_pred, y_prob)

    with open(result_path, "wb") as f:
        pickle.dump({"y_pred": y_pred, "y_prob": y_prob, "metrics": metrics}, f)


def _create_sequences_np(X, y, seq_len):
    n = X.shape[0] - seq_len + 1
    X_seq = np.empty((n, seq_len, X.shape[1]), dtype=X.dtype)
    y_seq = np.empty(n, dtype=y.dtype)
    for i in range(n):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len - 1]
    return X_seq, y_seq


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    return _create_sequences_np(X, y, seq_len)


class LSTMModel:
    pass


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
    payload = pickle.dumps(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "val_split": val_split,
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        result_path = f.name

    logger.info("    LSTM: launching subprocess...")
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_lstm_worker, args=(payload, result_path))
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise RuntimeError(f"LSTM subprocess failed with exit code {proc.exitcode}")

    with open(result_path, "rb") as f:
        result = pickle.load(f)

    import os

    os.unlink(result_path)

    return {
        "model": None,
        "y_pred": result["y_pred"],
        "y_prob": result["y_prob"],
        "metrics": result["metrics"],
    }
