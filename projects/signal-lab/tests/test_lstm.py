import numpy as np

from src.models.lstm import create_sequences, train_lstm


def _synthetic_data(n: int = 60, features: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    w = rng.normal(0, 1, features)
    prob = 1 / (1 + np.exp(-X @ w))
    y = (prob > 0.5).astype(int)
    split = n * 4 // 5
    return X[:split], y[:split], X[split:], y[split:]


def test_create_sequences_shape():
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    X_seq, y_seq = create_sequences(X, y, seq_len=10)
    assert X_seq.shape == (41, 10, 5)
    assert y_seq.shape == (41,)


def test_create_sequences_values():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    X_seq, y_seq = create_sequences(X, y, seq_len=3)
    np.testing.assert_array_equal(X_seq[0], X[:3])
    assert y_seq[0] == y[2]
    np.testing.assert_array_equal(X_seq[-1], X[7:10])
    assert y_seq[-1] == y[9]


def test_train_lstm_returns_required_keys():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_lstm(X_train, y_train, X_test, y_test, seq_len=5, max_epochs=5, patience=2)
    assert "model" in result
    assert "y_pred" in result
    assert "y_prob" in result
    assert "metrics" in result


def test_train_lstm_predictions_shape():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_lstm(X_train, y_train, X_test, y_test, seq_len=5, max_epochs=5, patience=2)
    n_expected = len(y_test) - 5 + 1
    assert len(result["y_pred"]) == n_expected
    assert len(result["y_prob"]) == n_expected


def test_train_lstm_probabilities_in_range():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_lstm(X_train, y_train, X_test, y_test, seq_len=5, max_epochs=5, patience=2)
    assert np.all(result["y_prob"] >= 0.0)
    assert np.all(result["y_prob"] <= 1.0)
