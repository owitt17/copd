import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def extract_features_with_windowing(file_path, max_pad_len=174, window_size=1.0, hop_size=0.5):
    """
    Extract MFCC features from audio files using time windowing.
    
    Parameters:
    - file_path: Path to the audio file.
    - max_pad_len: Maximum padding length for MFCC features.
    - window_size: Size of each window in seconds.
    - hop_size: Hop length (overlap) between consecutive windows in seconds.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        window_length = int(window_size * sample_rate)  # Number of samples per window
        hop_length = int(hop_size * sample_rate)       # Number of samples to hop

        # Split the audio into windows
        features = []
        for start in range(0, len(audio) - window_length + 1, hop_length):
            window_audio = audio[start:start + window_length]
            mfccs = librosa.feature.mfcc(y=window_audio, sr=sample_rate, n_mfcc=40)

            # Pad or truncate MFCC to a consistent size
            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
            
            features.append(mfccs)
        
        return np.array(features)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, Error: {e}")
        return None

# Set data paths
data_path = 'C:\\Users\\Owiit Gay\\Documents\\FYP production\\DATASETS'
labels = []
features = []

# Load data with time windowing
for label in ['HEALTHY', 'COPD']:
    folder = os.path.join(data_path, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        print(f"Processing file: {file_path}")
        mfcc_windows = extract_features_with_windowing(file_path)
        if mfcc_windows is not None:
            features.extend(mfcc_windows)  # Add all windows as individual samples
            labels.extend([1 if label == 'COPD' else 0] * len(mfcc_windows))  # Replicate labels for each window

# Convert lists to arrays
X = np.array(features)
y = np.array(labels)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape data for CNN input
X_train = X_train[..., np.newaxis]  # Add channel dimension for CNN
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define CNN model
model = Sequential([
    # Input Layer (e.g., Spectrogram or MFCC features)
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

# Evaluate model on test set
predictions = (model.predict(X_test) > 0.5).astype("int32")  # Threshold for binary classification
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("cnn_model_copd_windowed.h5")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_cnn_windowed.png')
plt.show()
