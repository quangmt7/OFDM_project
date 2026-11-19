import numpy as np


def apply_rayleigh_channel(
    ofdm_frames: np.ndarray,
    n_fft: int = 64,
    num_taps: int = 3,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pass OFDM frames through an independent 3-tap Rayleigh multipath channel per frame.
    Returns:
        faded_frames: shape (num_frames, frame_len)
        H_freq: Channel frequency response H[k], shape (num_frames, n_fft)
    """
    rng = np.random.default_rng(seed)
    num_frames, frame_len = ofdm_frames.shape

    # Complex Gaussian taps normalized so total average channel power = 1
    scale = np.sqrt(1.0 / (2.0 * num_taps))
    h_taps = scale * (
        rng.standard_normal((num_frames, num_taps))
        + 1j * rng.standard_normal((num_frames, num_taps))
    )

    faded_frames = np.zeros_like(ofdm_frames, dtype=np.complex128)
    for i in range(num_frames):
        # Linear convolution truncated to frame length
        conv_out = np.convolve(ofdm_frames[i], h_taps[i], mode="full")
        faded_frames[i] = conv_out[:frame_len]

    # Frequency-domain channel response H[k] = FFT(h, N_fft)
    H_freq = np.fft.fft(h_taps, n=n_fft, axis=1)

    return faded_frames, H_freq