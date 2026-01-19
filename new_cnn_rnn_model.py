import numpy as np
import librosa
import librosa.display
import os
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, BatchNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
spectrogram_folder = "spectrograms"
os.makedirs(spectrogram_folder, exist_ok=True)
MAX_TIME_STEPS = 128  # Fixed time length
N_MELS = 64  # Mel filter banks

# Load dataset
data_path = 'C:\\Users\\Owiit Gay\\Documents\\FYP production\\DATASETS'
labels, features = [], []

def save_spectrogram(file_path):
    """Extract fixed-size spectrogram"""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Pad or truncate
    if spectrogram_db.shape[1] < MAX_TIME_STEPS:
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, MAX_TIME_STEPS - spectrogram_db.shape[1])), mode='constant')
    else:
        spectrogram_db = spectrogram_db[:, :MAX_TIME_STEPS]

    return spectrogram_db

for label in ['HEALTHY', 'COPD']:
    folder = os.path.join(data_path, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            file_path = os.path.join(folder, file)
            spectrogram = save_spectrogram(file_path)
            features.append(spectrogram)
            labels.append(label)

# Convert to NumPy array
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

# Reshape for CNN
X = X[..., np.newaxis]  # (samples, n_mels, time_steps, 1)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define CNN-BiLSTM Model with L2 Regularization
model = Sequential([
    Reshape((N_MELS, MAX_TIME_STEPS, 1), input_shape=(N_MELS, MAX_TIME_STEPS, 1)),

    # CNN Feature Extractor
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Global Average Pooling (Instead of Flatten)
    GlobalAveragePooling2D(),

    # LSTM Sequence Processing
    Reshape((MAX_TIME_STEPS // 4, -1)),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(0.1))),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(0.1))),

    # Fully Connected Layers
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))

# Plot Accuracy and Loss
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
model.save("model_copd_improved.h5")
print("Model saved as model_copd_improved.h5")
