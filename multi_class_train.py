import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import glob

# read tất cả các file csv
csv_file = glob.glob('D:\Tai_lieu_hoc_tap2024-2025\TaiLieu_HK1-Nam3_2024-2025\Do_an_1\cicids\*.csv')

df_list = []

benign_samples = []
attack_samples = []

#lấy mẫu ngẫu nhiên từ mỗi tệp csv
for file in csv_file:

    df = pd.read_csv(file)

    # Lấy mẫu tấn công
    attack_filter = df[df.iloc[:, -1] != 'BENIGN']
    num_attack_samples = min(len(attack_filter), 5000)  # Lấy tối đa 5000 mẫu tấn công, nếu ít hơn 5000 thì láy attack_filter
    attack_samples.append(attack_filter.sample(n=num_attack_samples, random_state=42))

    # Lấy mẫu benign
    benign_filter = df[df.iloc[:, -1] == 'BENIGN']
    num_benign_samples = min(len(benign_filter), 5000)  # Lấy tối đa 5000 mẫu benign
    benign_samples.append(benign_filter.sample(n=num_benign_samples, random_state=42))

    print("số lượng benign: ",num_benign_samples)
    print("số lượng attack: ", num_attack_samples)


# Gộp tất cả các mẫu  của từng loại lại
benign_df = pd.concat(benign_samples, ignore_index=True)
attack_df = pd.concat(attack_samples, ignore_index=True)

# Gộp các mẫu lại với nhau
df = pd.concat([benign_df, attack_df], ignore_index=True)


attack_counts = df[df.iloc[:, -1] != 'BENIGN'].iloc[:, -1].value_counts()
# In kết quả ra màn hình
print("Số lượng từng loại tấn công:")
print(attack_counts)


# Giả sử cột cuối cùng là nhãn (Label), chia X (features) và y (target)
X = df.iloc[:, :-1].values  # Các cột đặc trưng
y = df.iloc[:, -1].values   # Cột nhãn

# Kiểm tra giá trị NaN và Infinity trong X
print("Có giá trị NaN trong X không?", np.isnan(X).any())
print("Có giá trị Infinity trong X không?", np.isinf(X).any())

# Thay thế giá trị NaN và Infinity (nếu có)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Mã hóa nhãn thành số nguyên
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chuyển nhãn thành dạng one-hot encoding
y_onehot = to_categorical(y_encoded)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Định hình lại dữ liệu cho LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Đầu ra đa lớp

learning_rate = 0.001  # Bạn có thể điều chỉnh giá trị này
optimizer = Adam(learning_rate=learning_rate)

# Compile mô hình
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')
print(f'Loss: {loss}')

# Lưu mô hình
model.save('multi_class_model.keras')




# Sử dụng mô hình đã huấn luyện để dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Chuyển dự đoán thành nhãn
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Giải mã nhãn dự đoán thành tên loại tấn công
predicted_attack_names = label_encoder.inverse_transform(y_pred_labels)

# Hiển thị nhãn thực tế và nhãn dự đoán
print("Nhãn thực tế và dự đoán:")
for i in range(10):  # In 10 dự đoán đầu tiên để kiểm tra
    print(f"Nhãn thực tế: {label_encoder.inverse_transform([y_test_labels[i]])[0]} - Dự đoán: {predicted_attack_names[i]}")







import matplotlib.pyplot as plt

# Vẽ biểu đồ độ chính xác
plt.figure(figsize=(12, 5))

# Độ chính xác trên tập huấn luyện và tập kiểm tra
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Vẽ biểu đồ tổn thất
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')



plt.tight_layout()
plt.show()
