import numpy as np


def lmmse_equalize(
    Y_freq: np.ndarray,
    H_freq: np.ndarray,
    n0: float,
    es: float = 1.0,
) -> np.ndarray:
    """Linear Minimum Mean Square Error (LMMSE) frequency-domain equalizer."""
    numerator = np.conj(H_freq)
    denominator = (np.abs(H_freq) ** 2) + (n0 / es)
    X_hat = (numerator / denominator) * Y_freq
    return X_hat.reshape(-1)


def zf_equalize(Y_freq: np.ndarray, H_freq: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Zero-Forcing (ZF) frequency-domain equalizer for comparison."""
    X_hat = Y_freq / (H_freq + eps)
    return X_hat.reshape(-1)