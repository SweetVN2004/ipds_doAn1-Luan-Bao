import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# Đọc dữ liệu đã lọc
df = pd.read_csv('20000_ddos_benign_train_sample.csv')

# Giả sử cột cuối cùng là nhãn (Label), chia X (features) và y (target)
X = df.iloc[:, :-1].values  # Các cột đặc trưng
y = df.iloc[:, -1].values   # Cột nhãn

# Kiểm tra giá trị NaN và Infinity trong X
print("Có giá trị NaN trong X không?", np.isnan(X).any())
print("Có giá trị Infinity trong X không?", np.isinf(X).any())

# Thay thế giá trị NaN và Infinity (nếu có)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# Chuyển nhãn thành giá trị số (0 cho BENIGN, 1 cho DDoS)
y = np.where(y == 'DDoS', 1, 0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Định hình lại dữ liệu cho LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])) #x_train_shape[0] là tổng số mẫu, [1] là số đặc trưng của mỗi mẫu
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='sigmoid'))  # Đầu ra nhị phân

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')
print(f'Loss: {loss}')
# Lưu mô hình
model.save('my_model.keras')

