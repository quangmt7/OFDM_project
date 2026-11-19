import numpy as np


def ofdm_modulate(symbols: np.ndarray, n_fft: int = 64, cp_len: int = 16) -> np.ndarray:
    """
    Convert 1D array of QAM symbols into OFDM time-domain frames with Cyclic Prefix.
    Returns shape: (num_ofdm_symbols, n_fft + cp_len)
    """
    if len(symbols) % n_fft != 0:
        raise ValueError(f"Number of symbols must be a multiple of n_fft ({n_fft}).")

    # Serial-to-Parallel
    freq_blocks = symbols.reshape(-1, n_fft)

    # Unitary IFFT (preserves average symbol power Es = 1)
    time_blocks = np.fft.ifft(freq_blocks, n=n_fft, axis=1, norm="ortho")

    # Add Cyclic Prefix (copy last cp_len samples to the front)
    cp = time_blocks[:, -cp_len:]
    ofdm_frames = np.hstack((cp, time_blocks))

    return ofdm_frames