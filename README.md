import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("nasa_cmaps.csv")  # Replace with actual path

# Assuming the last column is the target
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Scale features only
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Ensure labels are binary (0 and 1)
assert set(labels.unique()).issubset({0, 1}), "Labels must be binary (0 and 1)."

# Function to create sequences
def create_sequences(features, labels, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(features_scaled, labels.values, seq_length)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(100),
    Dropout(0.2),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_class))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()
