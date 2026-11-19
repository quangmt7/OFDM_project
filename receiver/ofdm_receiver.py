import numpy as np


def ofdm_demodulate(rx_frames: np.ndarray, n_fft: int = 64, cp_len: int = 16) -> np.ndarray:
    """
    Remove Cyclic Prefix and apply unitary FFT to recover frequency-domain symbols Y[k].
    Returns shape: (num_ofdm_symbols, n_fft)
    """
    # Strip Cyclic Prefix
    no_cp = rx_frames[:, cp_len : cp_len + n_fft]

    # Unitary FFT
    Y_freq = np.fft.fft(no_cp, n=n_fft, axis=1, norm="ortho")
    return Y_freq