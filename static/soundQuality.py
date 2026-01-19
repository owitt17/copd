import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

# Noise reduction using a bandpass filter
def bandpass_filter(audio, sr, lowcut=300, highcut=2500):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, audio)

# Remove heartbeat noise using a high-pass filter
def highpass_filter(audio, sr, cutoff=100):
    nyquist = 0.5 * sr
    cutoff_normalized = cutoff / nyquist
    b, a = butter(1, cutoff_normalized, btype='high')
    return lfilter(b, a, audio)

# Normalize audio
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# Simulate phone recording by resampling
def simulate_phone_quality(audio, original_sr, target_sr=8000):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr), target_sr

# Process audio files
def process_audio_files(input_dir, output_dir, filter_type):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):  # Process only .wav files
            file_path = os.path.join(input_dir, file_name)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                # Step 1: Apply the specified filter
                if filter_type == 'COPD':
                    filtered_audio = bandpass_filter(audio, sr)
                elif filter_type == 'HEALTHY':
                    filtered_audio = highpass_filter(audio, sr)
                else:
                    raise ValueError("Unknown filter type. Use 'COPD' or 'HEALTHY'.")

                # Step 2: Normalize
                normalized_audio = normalize_audio(filtered_audio)

                # Step 3: Simulate phone recording
                phone_audio, phone_sr = simulate_phone_quality(normalized_audio, sr)

                # Step 4: Save processed audio
                output_path = os.path.join(output_dir, file_name)
                sf.write(output_path, phone_audio, phone_sr)
                print(f"Processed {file_name} and saved to {output_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Directories
copd_input_dir = "DATASETS/COPD"
copd_output_dir = "NEW_COPD"
healthy_input_dir = "DATASETS/HEALTHY"
healthy_output_dir = "NEW_HEALTHY"

# Process COPD dataset
process_audio_files(copd_input_dir, copd_output_dir, filter_type='COPD')

# Process Healthy dataset
process_audio_files(healthy_input_dir, healthy_output_dir, filter_type='HEALTHY')
