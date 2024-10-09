import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential 
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Đọc dữ liệu đã lọc
dtrain = pd.read_csv('Dtrain.csv')
dtest = pd.read_csv('Dtest.csv')

# Giả sử cột cuối cùng là nhãn (Label), chia X (features) và y (target)
# Bộ train
X = dtrain.iloc[:, :-1].values  # Các cột đặc trưng
y = dtrain.iloc[:, -1].values   # Cột nhãn

# Bộ test
Xt = dtest.iloc[:, :-1].values  # Các cột đặc trưng
yt = dtest.iloc[:, -1].values   # Cột nhãn

# Thay thế giá trị NaN và Infinity (nếu có)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xt_scaled = scaler.fit_transform(Xt)

# Mã hóa nhãn thành số nguyên
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
yt_encoded = label_encoder.fit_transform(yt)

# Chuyển nhãn thành dạng one-hot encoding
y_onehot = to_categorical(y_encoded, num_classes=15)
yt_onehot = to_categorical(yt_encoded, num_classes=15)


# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = X_scaled, Xt_scaled, y_onehot, yt_onehot

# Định hình lại dữ liệu cho LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])) #x_train_shape[0] là tổng số mẫu, [1] là số đặc trưng của mỗi mẫu
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=15, activation='softmax')) 

# Compile mô hình\
optimizers = Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập test
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Loss: {loss}')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')

# Lưu mô hình
model.save('my_model.keras')

# Vẽ biểu đồ Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Vẽ biểu đồ Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

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
