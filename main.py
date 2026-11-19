import numpy as np

from transmitter.generate_bits import generate_bits
from transmitter.qam64 import modulate_64qam, demodulate_64qam
from transmitter.ofdm_transmitter import ofdm_modulate
from channel.rayleigh import apply_rayleigh_channel
from channel.awgn import add_awgn
from receiver.ofdm_receiver import ofdm_demodulate
from receiver.lmmse import lmmse_equalize, zf_equalize


def run_ofdm_pipeline(
    snr_db: float,
    num_ofdm_frames: int = 1000,
    n_fft: int = 64,
    cp_len: int = 16,
    num_taps: int = 3,
    bits_per_sym: int = 6,
) -> dict:
    """Run one end-to-end OFDM 64-QAM simulation pass at a given SNR (dB)."""
    total_bits = num_ofdm_frames * n_fft * bits_per_sym

    # 1. Transmitter
    tx_bits = generate_bits(total_bits)
    tx_symbols = modulate_64qam(tx_bits)
    tx_ofdm = ofdm_modulate(tx_symbols, n_fft=n_fft, cp_len=cp_len)

    # 2. Channel (3-tap Rayleigh + AWGN)
    faded_signal, H_freq = apply_rayleigh_channel(tx_ofdm, n_fft=n_fft, num_taps=num_taps)
    rx_signal, n0 = add_awgn(faded_signal, snr_db=snr_db)

    # 3. Receiver
    Y_freq = ofdm_demodulate(rx_signal, n_fft=n_fft, cp_len=cp_len)

    # LMMSE Equalization & Demodulation
    X_hat_lmmse = lmmse_equalize(Y_freq, H_freq, n0=n0, es=1.0)
    rx_bits_lmmse = demodulate_64qam(X_hat_lmmse)

    # ZF Equalization & Demodulation
    X_hat_zf = zf_equalize(Y_freq, H_freq)
    rx_bits_zf = demodulate_64qam(X_hat_zf)

    return {
        "snr_db": snr_db,
        "tx_bits": tx_bits,
        "rx_bits_lmmse": rx_bits_lmmse,
        "rx_bits_zf": rx_bits_zf,
    }


def main():
    snr_db_list = np.arange(0, 35, 5)

    for snr_db in snr_db_list:
        result = run_ofdm_pipeline(snr_db=snr_db)
        print(
            f"SNR = {result['snr_db']:2d} dB | "
            f"Transmitted {len(result['tx_bits']):,} bits | "
            f"LMMSE & ZF reception complete."
        )


if __name__ == "__main__":
    main()