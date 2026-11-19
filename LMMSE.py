import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1

np.random.seed(42)

# =========================================================================
# 1. 64-QAM MODULATION & DEMODULATION
# =========================================================================
PAM_MAP = np.array([-7, -5, -1, -3, +7, +5, +1, +3])
NORM_FACTOR = np.sqrt(42.0)

def modulate_64qam(bits):
    bits_reshaped = bits.reshape(-1, 6)
    i_idx = bits_reshaped[:, 0] * 4 + bits_reshaped[:, 1] * 2 + bits_reshaped[:, 2]
    q_idx = bits_reshaped[:, 3] * 4 + bits_reshaped[:, 4] * 2 + bits_reshaped[:, 5]
    return (PAM_MAP[i_idx] + 1j * PAM_MAP[q_idx]) / NORM_FACTOR

def demodulate_64qam(rx_symbols):
    flat_rx = rx_symbols.ravel()
    unnorm = flat_rx * NORM_FACTOR
    r_I, r_Q = np.real(unnorm), np.imag(unnorm)
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    
    det_I = levels[np.argmin(np.abs(r_I[:, None] - levels), axis=1)]
    det_Q = levels[np.argmin(np.abs(r_Q[:, None] - levels), axis=1)]
    det_symbols = ((det_I + 1j * det_Q) / NORM_FACTOR).reshape(rx_symbols.shape)
    
    gray_lut = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
    ], dtype=int)
    rx_bits = np.hstack([gray_lut[(det_I + 7) // 2], gray_lut[(det_Q + 7) // 2]]).ravel()
    return rx_bits, det_symbols

# =========================================================================
# 2. OFDM & RAYLEIGH MULTIPATH CHANNEL
# =========================================================================
def generate_rayleigh_channel(n_frames, N, L=8, beta=2.0):
    l = np.arange(L)
    p = np.exp(-l / beta)
    p = p / np.sum(p)
    h = np.sqrt(p / 2.0) * (np.random.randn(n_frames, L) + 1j * np.random.randn(n_frames, L))
    H_freq = np.fft.fft(h, n=N, axis=-1)
    return h, H_freq

def transmit_ofdm_rayleigh_awgn(X_frames, h, N_cp, snr_linear):
    n_frames, N = X_frames.shape
    L = h.shape[1]
    x_time = np.fft.ifft(X_frames, axis=-1)
    x_cp = np.concatenate([x_time[:, -N_cp:], x_time], axis=-1)
    
    y_cp = np.zeros_like(x_cp)
    for l in range(L):
        y_cp[:, l:] += h[:, l:l+1] * x_cp[:, :N + N_cp - l]
        
    N0 = 1.0 / snr_linear
    noise_std = np.sqrt(N0 / (2.0 * N))
    w_cp = noise_std * (np.random.randn(*y_cp.shape) + 1j * np.random.randn(*y_cp.shape))
    
    y_no_cp = (y_cp + w_cp)[:, N_cp:N_cp + N]
    return np.fft.fft(y_no_cp, axis=-1)

# =========================================================================
# 3. LMMSE EQUALIZER MODULE
# =========================================================================
def equalize_lmmse(Y, H, snr_linear):
    H_abs_sq = np.abs(H) ** 2
    inv_snr = 1.0 / snr_linear
    G_lmmse = np.conj(H) / (H_abs_sq + inv_snr)
    alpha = H_abs_sq / (H_abs_sq + inv_snr)
    
    X_hat_biased = G_lmmse * Y
    X_hat_unbiased = X_hat_biased / alpha
    return X_hat_biased, X_hat_unbiased

# =========================================================================
# 4. MONTE CARLO SIMULATION SWEEP (SNR = 0 to 40 dB)
# =========================================================================
N, N_cp, L = 64, 16, 8
N_frames = 3000  # 192,000 symbols (1,152,000 bits) per SNR point
M, k_bits = 64, 6
snr_dB_range = np.arange(0, 42, 4)

mse_lmmse_sim, mse_lmmse_theo = [], []
ber_lmmse_biased, ber_lmmse_unbiased = [], []
ser_lmmse_biased, ser_lmmse_unbiased = [], []
ber_rayleigh_theo, ser_rayleigh_theo = [], []

for snr_dB in snr_dB_range:
    gamma = 10.0 ** (snr_dB / 10.0)
    tx_bits = np.random.randint(0, 2, N_frames * N * k_bits)
    X_frames = modulate_64qam(tx_bits).reshape(N_frames, N)
    
    h, H_freq = generate_rayleigh_channel(N_frames, N, L=L, beta=2.0)
    Y_freq = transmit_ofdm_rayleigh_awgn(X_frames, h, N_cp, gamma)
    
    # LMMSE Equalization
    X_lmmse_b, X_lmmse_u = equalize_lmmse(Y_freq, H_freq, gamma)
    
    # Result 3: LMMSE MSE vs. SNR
    mse_lmmse_sim.append(np.mean(np.abs(X_frames - X_lmmse_b) ** 2))
    inv_g = 1.0 / gamma
    mse_lmmse_theo.append(inv_g * np.exp(inv_g) * exp1(inv_g))
    
    # Result 4: LMMSE BER & SER vs. SNR
    bits_lb, sym_lb = demodulate_64qam(X_lmmse_b)
    bits_lu, sym_lu = demodulate_64qam(X_lmmse_u)
    
    ber_lmmse_biased.append(np.mean(tx_bits != bits_lb))
    ber_lmmse_unbiased.append(np.mean(tx_bits != bits_lu))
    ser_lmmse_biased.append(np.mean(np.abs(X_frames - sym_lb) > 1e-5))
    ser_lmmse_unbiased.append(np.mean(np.abs(X_frames - sym_lu) > 1e-5))
    
    a = 1.0 - 1.0 / np.sqrt(M)
    b = (1.5 / (M - 1)) * gamma
    mu = np.sqrt(b / (1.0 + b))
    ser_th = 2 * a * (1.0 - mu) - (a ** 2) * (1.0 - (4.0 / np.pi) * mu * np.arctan(1.0 / mu))
    ser_rayleigh_theo.append(ser_th)
    ber_rayleigh_theo.append(ser_th / k_bits)

# =========================================================================
# 5. PLOT RESULT 3 (LMMSE MSE) & RESULT 4 (LMMSE BER/SER)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.semilogy(snr_dB_range, mse_lmmse_sim, 'bs-', lw=1.8, ms=6, label='LMMSE MSE (Simulated)')
ax1.semilogy(snr_dB_range, mse_lmmse_theo, 'k:', lw=2.2, label='LMMSE MSE (Theory)')
ax1.set(xlabel='SNR = Es/N0 (dB)', ylabel='Mean Square Error (MSE)',
        title='LMMSE Equalizer MSE vs. SNR over Rayleigh Fading')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', fontsize=9)

ax2.semilogy(snr_dB_range, ser_lmmse_biased, 'c^--', lw=1.5, ms=5, label='Simulated SER (Biased LMMSE)')
ax2.semilogy(snr_dB_range, ser_lmmse_unbiased, 'g^-', lw=1.8, ms=6, label='Simulated SER (Unbiased LMMSE)')
ax2.semilogy(snr_dB_range, ser_rayleigh_theo, 'k--', lw=1.5, label='Theoretical Rayleigh SER')
ax2.semilogy(snr_dB_range, ber_lmmse_biased, 'ms--', lw=1.5, ms=5, label='Simulated BER (Biased LMMSE)')
ax2.semilogy(snr_dB_range, ber_lmmse_unbiased, 'bo-', lw=1.8, ms=6, label='Simulated BER (Unbiased LMMSE)')
ax2.semilogy(snr_dB_range, ber_rayleigh_theo, 'r:', lw=2.0, label='Theoretical Rayleigh BER (Gray)')
ax2.set(ylim=[1e-4, 1], xlabel='SNR = Es/N0 (dB)', ylabel='Error Rate (BER / SER)',
        title='64-QAM BER & SER vs. SNR with LMMSE Equalizer')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(loc='lower left', fontsize=8.5)

plt.tight_layout()
plt.show()