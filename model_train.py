import numpy as np
import librosa
import os
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None

# Dataset path
data_path = 'C:\\Users\\Owiit Gay\\Documents\\FYP production\\DATASETS'
labels, features = [], []

# Load data
for label in ['HEALTHY', 'COPD']:
    folder = os.path.join(data_path, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            file_path = os.path.join(folder, file)
            mfcc = extract_features(file_path)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(label)

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

print(le.classes_)

joblib.dump(le, 'label_encoder.pkl')

# Load and apply the same encoder in testing
le = joblib.load('label_encoder.pkl')

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Label distribution in training set: {np.bincount(y_train)}")
print(f"Label distribution in validation set: {np.bincount(y_val)}")
print(f"Label distribution in testing set: {np.bincount(y_test)}")

# Model
model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh', return_sequences=True, kernel_regularizer=l2(0.1)),
    Dropout(0.4),
    LSTM(32, activation='tanh', kernel_regularizer=l2(0.1)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Training
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_val, y_val))

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("training_performance.png")
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save model
model.save("model_copd.h5")
print("Model saved as model_copd.h5")
