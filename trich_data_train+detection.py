import pandas as pd

# Đọc file dữ liệu lớn ban đầu
df = pd.read_csv('D:\\Tai_lieu_hoc_tap2024-2025\\TaiLieu_HK1-Nam3_2024-2025\\Do_an_1\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Lọc các dòng có nhãn 'DDoS' và 'BENIGN'
#df[...]: Khi bạn truyền một Series Boolean vào DataFrame, Pandas sẽ trả về một DataFrame mới chỉ chứa các dòng mà điều kiện là True.

df_ddos = df[df[' Label'] == 'DDoS'] # df[boolean], df[' label' == 'DDoS'] la dieu kien. 
df_benign = df[df[' Label'] == 'BENIGN']

# Lấy mẫu 20000 dòng từ mỗi loại (DDoS và BENIGN) cho train
df_ddos_train_sample = df_ddos.sample(20000, random_state=42)
df_benign_train_sample = df_benign.sample(20000, random_state=42)

# Kết hợp lại thành một dataset cho train
df_train_sample = pd.concat([df_ddos_train_sample, df_benign_train_sample])

# Lưu lại dữ liệu đã lọc cho train
df_train_sample.to_csv('20000_ddos_benign_train_sample.csv', index=False)

# Tạo dataset cho detect bằng cách loại bỏ các mẫu đã lấy ở trên từ dataset lớn
df_ddos_detect = df_ddos.drop(df_ddos_train_sample.index)
df_benign_detect = df_benign.drop(df_benign_train_sample.index)

# Lấy mẫu từ dataset còn lại (detect)
df_ddos_detect_sample = df_ddos_detect.sample(20000, random_state=42)
df_benign_detect_sample = df_benign_detect.sample(20000, random_state=42)

# Kết hợp lại thành một dataset cho detect
df_detect_sample = pd.concat([df_ddos_detect_sample, df_benign_detect_sample])

# Lưu lại dữ liệu đã lọc cho detect
df_detect_sample.to_csv('20000_ddos_benign_detect_sample.csv', index=False)

# Xem dữ liệu đã lọc
print("Train Dataset:")
print(df_train_sample.head())
print("\nDetect Dataset:")
print(df_detect_sample.head())
