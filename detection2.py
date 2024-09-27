import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#numpy config
# np.set_printoptions(threshold=np.inf)  # Hiển thị toàn bộ mảng list_predict 
np.set_printoptions(suppress=True, precision=6)  # Suppress sẽ không hiển thị số mũ và precision là số chữ số thập phân

# Đọc dữ liệu kiểm tra hoặc dữ liệu mới
new_data = pd.read_csv('20000_ddos_benign_detect_sample.csv')

# Giả sử cột cuối cùng là nhãn (Label), chia X (features) và y (target)
feature_data = new_data.iloc[:, :-1].values  # Các cột đặc trưng
target_data = new_data.iloc[:, -1].values   # Cột nhãn (nếu có)

# Xử lý giá trị NaN và vô cực
feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

# Chuyển nhãn thành giá trị số (0 cho BENIGN, 1 cho DDoS)
target_data = np.where(target_data == 'DDoS', 1, 0)

# Chuẩn hóa dữ liệu mới
scaler = StandardScaler()
feature_data_scaled = scaler.fit_transform(feature_data)

# Định hình lại dữ liệu cho LSTM (samples, timesteps, features)
feature_data_reshaped = np.reshape(feature_data_scaled, (feature_data_scaled.shape[0], 1, feature_data_scaled.shape[1]))

# Tải mô hình đã được huấn luyện
model = load_model('my_model.keras')

# Dự đoán với dữ liệu mới
list_predict = model.predict(feature_data_reshaped)

# Chuyển đổi giá trị dự đoán từ xác suất sang nhãn nhị phân
list_predict_classed = (list_predict > 0.5).astype(int)
# print(list_predict)
# Chuyển đổi mảng thành danh sách
list_predict_convert = list_predict.tolist()

# In từng phần tử với dấu phần trăm
# for value in list_predict_convert:
#     print(f"{value[0] * 100:.2f}%")  # In giá trị dưới dạng phần trăm với 2 chữ số thập phân


# Chuyển đổi thành DataFrame với phần trăm
df = pd.DataFrame({
    'Predicted Probability (%)': list_predict.flatten() * 100,
    'Predicted Label': np.where(list_predict_classed.flatten() == 1, 'DDoS', 'BENIGN')
})
# In DataFrame
print(df)


# Nếu bạn có nhãn thực tế, đánh giá mô hình
if 'target_data' in locals():
    accuracy = accuracy_score(target_data, list_predict_classed)
    print(f'Accuracy on new data: {accuracy * 100:.2f}%')
    print("Classification Report (New Data):")
    print(classification_report(target_data, list_predict_classed))
else:
    print("Dự đoán thành công, không có nhãn thực tế để đánh giá.")

# # Hiển thị một mẫu ngẫu nhiên và dự đoán của nó
# sample_index = np.random.randint(0, X_new_scaled.shape[0])  # Chọn một chỉ số ngẫu nhiên
# sample_data = X_new[sample_index]  # Dữ liệu mẫu
# predicted_class = y_new_pred_classes[sample_index][0]  # Dự đoán cho mẫu

# # Chuyển đổi nhãn dự đoán thành dạng có thể đọc được
# predicted_label = 'DDoS' if predicted_class == 1 else 'BENIGN'

# # print("\nSample Data:")
# # print(sample_data)
# print(f"\nPredicted Class: {predicted_label}")
