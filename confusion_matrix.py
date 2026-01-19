import numpy as np
import librosa
import os
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def extract_features(file_path):
    """Extract mel spectrogram features to match training input shape (64, 128)."""
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Ensure fixed size (64, 128)
        if spectrogram_db.shape[1] < 128:
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, 128 - spectrogram_db.shape[1])), mode='constant')
        else:
            spectrogram_db = spectrogram_db[:, :128]

        return spectrogram_db
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None

# Load the trained model
model = tf.keras.models.load_model("model_copd_improved.h5")
print("Model loaded successfully")

# Load the label encoder
le = joblib.load('label_encoder.pkl')
print("Label Encoder loaded successfully")

# Load new dataset and perform predictions for confusion matrix
new_data_path = 'C:\\Users\\Owiit Gay\\Documents\\FYP production\\NEW_DATASETS'
new_labels, new_features = [], []

for label in ['HEALTHY', 'COPD']:
    folder = os.path.join(new_data_path, label)
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                file_path = os.path.join(folder, file)
                mfcc = extract_features(file_path)
                if mfcc is not None:
                   new_features.append(mfcc)
                   new_labels.append(label)


X_new = np.array(new_features)
y_new = np.array(new_labels)

y_new = le.transform(y_new) 

X_new = X_new[..., np.newaxis] 

predictions = model.predict(X_new)
predicted_classes = (predictions > 0.5).astype(int)

# Generate and display the confusion matrix
cm = confusion_matrix(y_new, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for New Dataset')
plt.savefig("confusion_matrix_new_dataset.png")
plt.show()