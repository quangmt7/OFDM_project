import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float, seed: int = None) -> tuple[np.ndarray, float]:
    """
    Add complex AWGN to the signal assuming unit average signal power (Es = 1).
    Returns:
        noisy_signal: signal + noise
        n0: noise variance (N0)
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10.0 ** (snr_db / 10.0)
    n0 = 1.0 / snr_linear

    noise = np.sqrt(n0 / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return signal + noise, n0