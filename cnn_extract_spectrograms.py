import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths
dataset_path = "DATASETS"
spectrogram_path = "spectrograms"
copd_path = os.path.join(dataset_path, "COPD")
healthy_path = os.path.join(dataset_path, "HEALTHY")

# Ensure spectrogram directories exist
os.makedirs(os.path.join(spectrogram_path, "COPD"), exist_ok=True)
os.makedirs(os.path.join(spectrogram_path, "HEALTHY"), exist_ok=True)

# Function to generate and save spectrograms
def save_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)  # Load audio
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)  # Convert to decibel scale
    
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram: {os.path.basename(audio_path)}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Get sample files (one COPD, one Healthy)
copd_files = [f for f in os.listdir(copd_path) if f.endswith('.wav')]
healthy_files = [f for f in os.listdir(healthy_path) if f.endswith('.wav')]

# Process one file from each category
if copd_files and healthy_files:
    save_spectrogram(os.path.join(copd_path, copd_files[0]), os.path.join(spectrogram_path, "COPD", "copd_sample.png"))
    save_spectrogram(os.path.join(healthy_path, healthy_files[0]), os.path.join(spectrogram_path, "HEALTHY", "healthy_sample.png"))

# Generate a combined spectrogram
if copd_files and healthy_files:
    y_copd, sr_copd = librosa.load(os.path.join(copd_path, copd_files[0]), sr=None)
    y_healthy, sr_healthy = librosa.load(os.path.join(healthy_path, healthy_files[0]), sr=None)

    # Combine signals
    min_length = min(len(y_copd), len(y_healthy))
    y_combined = np.hstack((y_copd[:min_length], y_healthy[:min_length]))

    S_combined = librosa.feature.melspectrogram(y=y_combined, sr=sr_copd)
    S_combined_db = librosa.power_to_db(S_combined, ref=np.max)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_combined_db, sr=sr_copd, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Combined COPD & Healthy Spectrogram")
    plt.savefig(os.path.join(spectrogram_path, "combined_sample.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("Spectrogram images saved in:", spectrogram_path)

# ---------------------
# CNN Model for Feature Extraction
# ---------------------

# Create a simple CNN model for spectrogram classification
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Two classes: COPD and Healthy
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model structure
model.save("cnn_feature_extractor.h5")

print("CNN model saved as cnn_feature_extractor.h5")
