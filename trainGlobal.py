import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.metrics import Precision, Recall # type: ignore
import matplotlib.pyplot as plt

# Đọc dữ liệu đã lọc
dtrain = pd.read_csv('Dtrain1.csv')
dtest = pd.read_csv('Dtest1.csv')

# Giả sử cột cuối cùng là nhãn (Label), chia X (features) và y (target)
# Bộ train
X = dtrain.iloc[:, :-1].values  # Các cột đặc trưng
y = dtrain.iloc[:, -1].values   # Cột nhãn

# Bộ test
Xt = dtest.iloc[:, :-1].values  # Các cột đặc trưng
yt = dtest.iloc[:, -1].values   # Cột nhãn

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xt_scaled = scaler.transform(Xt)

# Mã hóa nhãn thành số nguyên
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
yt_encoded = label_encoder.fit_transform(yt)

# Chuyển nhãn thành dạng one-hot encoding
y_onehot = to_categorical(y_encoded, num_classes=15)
yt_onehot = to_categorical(yt_encoded, num_classes=15)

# Chia tập dữ liệu thành train và test (sau khi áp dụng SMOTE cho X_train, y_train)
X_train, X_test, y_train, y_test = X_scaled, Xt_scaled, y_onehot, yt_onehot

# Định hình lại dữ liệu cho LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))  # reshape sau khi SMOTE hoặc class_weight
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(GRU(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(units=64))
# Thêm lớp Dropout để tránh overfitting
model.add(Dropout(0.2))
model.add(Dense(units=15, activation='softmax'))  

from tensorflow.keras.optimizers import AdamW # type: ignore
# Compile mô hình
optimizers = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# # Định nghĩa callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Huấn luyện mô hình với class_weight
history = model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=64, 
    validation_data=(X_test, y_test))
    # , callbacks=[early_stopping, reduce_lr])                                 # batch size nhỏ các trọng số cập nhật nhanh --->chỉ số không ổn định

# Đánh giá mô hình trên tập test
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Loss: {loss}')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')

# Lưu mô hình
model.save('my_model.keras')


# Tạo figure và chia làm 2 subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Vẽ biểu đồ Accuracy
axs[0].plot(history.history['accuracy'], label='Train Accuracy')
axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(loc='upper left')

# Vẽ biểu đồ Loss
axs[1].plot(history.history['loss'], label='Train Loss')
axs[1].plot(history.history['val_loss'], label='Validation Loss')
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper left')

# Hiển thị cả hai biểu đồ cùng lúc
plt.tight_layout()
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
for i in range(100):  # In 10 dự đoán đầu tiên để kiểm tra
    print(f"Nhãn thực tế: {label_encoder.inverse_transform([y_test_labels[i]])[0]} - Dự đoán: {predicted_attack_names[i]}")
