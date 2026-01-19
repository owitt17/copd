import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     BatchNormalization, Input, LSTM, Bidirectional, 
                                     TimeDistributed, GaussianNoise, Reshape)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from librosa import effects
from sklearn.utils import class_weight

# Function to denoise audio
def denoise_audio(audio, sr):
    denoised = effects.preemphasis(audio)
    y_h = librosa.effects.hpss(denoised)[0]
    return y_h

# Function to augment audio with respiratory sounds
def augment_audio(features, sr):
    try:
        n_features, time_steps = features.shape
        audio = features.reshape(-1)
        rate_change = np.random.uniform(low=0.9, high=1.1)
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate_change)
        n_steps = np.random.randint(-2, 3)
        shifted_audio = librosa.effects.pitch_shift(stretched_audio, sr=sr, n_steps=n_steps)
        augmented = shifted_audio[:n_features * time_steps].reshape(n_features, time_steps)
        if augmented.shape != features.shape:
            augmented = np.pad(augmented, 
                             ((0, 0), (0, features.shape[1] - augmented.shape[1])), 
                             mode='constant')
            augmented = augmented[:, :features.shape[1]]
        return augmented
    except Exception as e:
        print(f"Error in augmentation: {e}")
        return features

# Function to extract features with time windowing
def extract_features_with_windowing(file_path, max_pad_len=174, window_size=1.0, hop_size=0.5, augment=False):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        audio = denoise_audio(audio, sample_rate)
        if augment:
            audio = augment_audio(audio, sample_rate)
        window_length = int(window_size * sample_rate)
        hop_length = int(hop_size * sample_rate)
        features = []
        for start in range(0, len(audio) - window_length + 1, hop_length):
            window_audio = audio[start:start + window_length]
            mfccs = librosa.feature.mfcc(y=window_audio, sr=sample_rate, n_mfcc=40,
                                       fmin=100, fmax=2000)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=window_audio, sr=sample_rate)
            spectral_centroid = librosa.feature.spectral_centroid(y=window_audio, sr=sample_rate)
            combined_features = np.vstack([mfccs, spectral_rolloff, spectral_centroid])
            if combined_features.shape[1] < max_pad_len:
                pad_width = max_pad_len - combined_features.shape[1]
                combined_features = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                combined_features = combined_features[:, :max_pad_len]
            features.append(combined_features)
        return np.array(features)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, Error: {e}")
        return None

# Set paths
data_path = 'C:\\Users\\Owiit Gay\\Documents\\FYP production\\DATASETS'
labels = []
features = []

# Load data with time windowing
for label in ['HEALTHY', 'COPD']:
    folder = os.path.join(data_path, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        mfcc_windows = extract_features_with_windowing(file_path)
        if mfcc_windows is not None:
            features.extend(mfcc_windows)
            labels.extend([1 if label == 'COPD' else 0] * len(mfcc_windows))

X_temp = np.array(features)
y_temp = np.array(labels)

# Split data
X_train, X_temp2, y_train, y_temp2 = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp2, y_temp2, test_size=0.5, random_state=42)

# Augment COPD samples to balance dataset
features_train = []
labels_train = []
for i in range(len(X_train)):
    features_train.append(X_train[i])
    labels_train.append(y_train[i])
    if y_train[i] == 1:
        augmented_feature = augment_audio(X_train[i], sr=22050)
        if augmented_feature is not None and augmented_feature.shape == X_train[i].shape:
            features_train.append(augmented_feature)
            labels_train.append(1)

X_train = np.array(features_train)
y_train = np.array(labels_train)

# Normalize data
mean = np.mean(X_train, axis=(0,1,2))
std = np.std(X_train, axis=(0,1,2))

X_train = (X_train-mean)/std
X_val = (X_val-mean)/std
X_test = (X_test-mean)/std

# Reshape data for CNN input
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# CNN-RNN Hybrid Model with adjustments
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2], 1))

# Add Gaussian noise layer
noise_layer = GaussianNoise(0.1)(input_layer)

# CNN Block with stronger regularization
cnn = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', 
             kernel_regularizer=l2(0.04))(noise_layer)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.5)(cnn)

cnn = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
             kernel_regularizer=l2(0.04))(cnn)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.5)(cnn)

cnn = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
             kernel_regularizer=l2(0.04))(cnn)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.5)(cnn)

# Flatten CNN output
flattened = Flatten()(cnn)

# Reshape for RNN
cnn_output_shape = cnn.shape[1:]
rnn_input_width = cnn_output_shape[2]
rnn_input_length = tf.keras.backend.int_shape(flattened)[1] // rnn_input_width
reshaped = Reshape((rnn_input_length, rnn_input_width))(flattened)

# RNN Block
rnn = Bidirectional(LSTM(64, return_sequences=True, 
                        kernel_regularizer=l2(0.04),
                        recurrent_regularizer=l2(0.04)))(reshaped)
rnn = Dropout(0.5)(rnn)
rnn = Bidirectional(LSTM(32, return_sequences=False,
                        kernel_regularizer=l2(0.04),
                        recurrent_regularizer=l2(0.04)))(rnn)
rnn = Dropout(0.5)(rnn)

# Output layer
output = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.04))(rnn)

# Create and compile model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(y_train),
                                                y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Train model
history = model.fit(X_train, y_train,
                   epochs=100,
                   batch_size=32,
                   validation_data=(X_val, y_val),
                   class_weight=class_weight_dict,
                   callbacks=[early_stopping, reduce_lr])

# Save model
model.save("cnn_rnn_model_copd.h5")

# Evaluate model
predictions = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

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
plt.savefig('cnn_rnn_training_history.png')
plt.show()
