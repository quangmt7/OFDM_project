import numpy as np

# 3-bit Gray-coded PAM-8 mapping for I and Q axes
GRAY_3BIT_TO_PAM8 = {
    (0, 0, 0): -7,
    (0, 0, 1): -5,
    (0, 1, 1): -3,
    (0, 1, 0): -1,
    (1, 1, 0):  1,
    (1, 1, 1):  3,
    (1, 0, 1):  5,
    (1, 0, 0):  7,
}

PAM8_LEVELS = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
PAM8_TO_GRAY_3BIT = {v: k for k, v in GRAY_3BIT_TO_PAM8.items()}

# Normalization factor for 64-QAM so that E[|X|^2] = Es = 1
NORM_FACTOR = np.sqrt(42.0)


def modulate_64qam(bits: np.ndarray) -> np.ndarray:
    """Map binary array (length multiple of 6) to normalized 64-QAM complex symbols."""
    if len(bits) % 6 != 0:
        raise ValueError("Number of bits must be a multiple of 6 for 64-QAM.")

    bit_groups = bits.reshape(-1, 6)
    i_vals = np.array([GRAY_3BIT_TO_PAM8[tuple(b[:3])] for b in bit_groups], dtype=np.float64)
    q_vals = np.array([GRAY_3BIT_TO_PAM8[tuple(b[3:])] for b in bit_groups], dtype=np.float64)

    return (i_vals + 1j * q_vals) / NORM_FACTOR


def demodulate_64qam(symbols: np.ndarray) -> np.ndarray:
    """Demodulate 64-QAM complex symbols back to a binary array using hard decision."""
    scaled = symbols * NORM_FACTOR
    i_rx = np.real(scaled)
    q_rx = np.imag(scaled)

    # Nearest-neighbor decision on I and Q axes independently
    i_idx = np.argmin(np.abs(i_rx[:, None] - PAM8_LEVELS[None, :]), axis=1)
    q_idx = np.argmin(np.abs(q_rx[:, None] - PAM8_LEVELS[None, :]), axis=1)

    bits_out = np.empty((len(symbols), 6), dtype=np.int8)
    for idx in range(len(symbols)):
        bits_out[idx, :3] = PAM8_TO_GRAY_3BIT[PAM8_LEVELS[i_idx[idx]]]
        bits_out[idx, 3:] = PAM8_TO_GRAY_3BIT[PAM8_LEVELS[q_idx[idx]]]

    return bits_out.reshape(-1)