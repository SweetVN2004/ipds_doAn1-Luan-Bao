import pandas as pd

# Đọc file dữ liệu
df = pd.read_csv('D:\Tai_lieu_hoc_tap2024-2025\TaiLieu_HK1-Nam3_2024-2025\Do_an_1\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Lọc các dòng có nhãn 'DDoS' và 'BENIGN'
#df[...]: Khi bạn truyền một Series Boolean vào DataFrame, Pandas sẽ trả về một DataFrame mới chỉ chứa các dòng mà điều kiện là True.
df_ddos = df[df[' Label'] == 'DDoS'] #df[boolean] - df[' label' == 'DDoS'] la dieu kien. 
df_benign = df[df[' Label'] == 'BENIGN']

# Lấy mẫu 1000 dòng từ mỗi loại (DDoS và BENIGN)
df_ddos_sample = df_ddos.sample(20000)
df_benign_sample = df_benign.sample(20000)

# Kết hợp cả hai loại lại thành một dataset
df_sample = pd.concat([df_ddos_sample, df_benign_sample])

# Xem dữ liệu đã lọc
print(df_sample.head())

# Lưu lại dữ liệu đã lọc
df_sample.to_csv('20000_ddos_benign_sample.csv', index=False)
