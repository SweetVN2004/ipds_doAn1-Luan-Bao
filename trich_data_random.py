import pandas as pd

# Đọc file dữ liệu
df = pd.read_csv(r'D:\Tai_lieu_hoc_tap2024-2025\TaiLieu_HK1-Nam3_2024-2025\Do_an_1\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Lọc các dòng có nhãn 'DDoS' và 'BENIGN'
df_ddos = df[df[' Label'] == 'DDoS']
df_benign = df[df[' Label'] == 'BENIGN']

# Kiểm tra số lượng dòng có trong mỗi loại
print(f"Số dòng DDoS: {len(df_ddos)}")
print(f"Số dòng BENIGN: {len(df_benign)}")

# Số mẫu bạn muốn lấy
sample_size = 10

# Lấy mẫu từ từng loại (nếu số dòng trong loại đó đủ)
df_ddos_sample = df_ddos.sample(min(sample_size, len(df_ddos)), random_state=42)
df_benign_sample = df_benign.sample(min(sample_size, len(df_benign)), random_state=42)

# Kết hợp cả hai loại lại thành một dataset
df_sample = pd.concat([df_ddos_sample, df_benign_sample])

# Xem dữ liệu đã lọc
print(df_sample.head())

# Lưu lại dữ liệu đã lọc
df_sample.to_csv('10_ddos_benign_sample.csv', index=False)
