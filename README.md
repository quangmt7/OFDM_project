# OFDM_project
Mô phỏng hệ thống OFDM sử dụng bộ lọc LMMSE trên kênh truyền Rayleigh nhiều đường, điều chế 64-QAM. Đánh giá chất lượng của BER/SER.
main.py : File chạy chính
transmitter/
+generate_bits.py: Tạo bit nhị phân
+qam64.py : Điều chế 64-QAM
+ofdm_transmitter.py: Sắp xếp OFDM, IFFT và thêm CP

  channel/
+rayleigh.py: Mô hình fading
+awgn.py: Nhiễu trắng

  receiver/
+ofdm_receiver.py: Bỏ OFDM, thực hiện FFT
+lmmse.py: Cân bằng kênh LMMSE

evaluation/
+ber_ser.py: Tính BER và SER
+plot_results.py: Vẽ đồ thị theo SNR

results/
+ber_results.csv: Số liệu của BER
+ser_results.csv: Số liệu của SER
+BER_vs_SNR.png: Ảnh so sánh BER với SNR 
+SER_vs_SNR.png: Ảnh so sánh SER với SNR 
