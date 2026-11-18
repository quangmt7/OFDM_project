# OFDM_project
Mô phỏng hệ thống OFDM sử dụng bộ lọc LMMSE trên kênh truyền Rayleigh nhiều đường, điều chế 64-QAM. Đánh giá chất lượng của BER/SER.
OFDM_LMMSE_Project/
│
├── main.py
│
├── transmitter/
│   ├── generate_bits.py
│   ├── qam64.py
│   └── ofdm_transmitter.py
│
├── channel/
│   ├── rayleigh.py
│   └── awgn.py
│
├── receiver/
│   ├── ofdm_receiver.py
│   └── lmmse.py
│
├── evaluation/
│   ├── ber_ser.py
│   └── plot_results.py
│
├── results/
│
└── report/
