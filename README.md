# COPD Audio Classification Web App

This project is a Flask web app that uses deep learning to classify user-submitted breathing audio as either COPD or Healthy. The app loads a trained CNN/RNN model for audio classification and provides a web interface for upload and results. It also includes a COPD-focused chatbot powered by the Gemini API.

## What I Used

- Python + Flask for the web server and API
- TensorFlow / Keras for model inference
- CNN + RNN architecture for audio classification
- Librosa for audio feature extraction (MFCCs)
- Gemini API for the COPD-only chatbot

## How It Works (High Level)

1. User uploads a `.wav` breathing audio file.
2. The server extracts MFCC features with Librosa.
3. The trained CNN/RNN model predicts COPD vs Healthy.
4. The result is returned to the UI.

## Project Structure (Key Files)

- `app.py` - Flask app, API routes, model loading, inference
- `templates/` - HTML pages
- `static/` - CSS/JS assets
- `model_copd.h5` - Trained model used for prediction
- `uploads/` - Temporary upload folder (files are deleted after prediction)

## Setup and Run

### 1) Install Python (macOS example)

```
brew install python@3.11
```

### 2) Create and activate a virtual environment

```
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```
pip install flask numpy librosa google-generativeai resampy
pip install tensorflow-macos tensorflow-metal
```

If you are not on Apple Silicon, try:

```
pip install tensorflow
```

### 4) Run the app

```
python app.py
```

Then open:

```
http://127.0.0.1:5000/
```

## Notes

- Only `.wav` files are supported for upload.
- The chatbot uses the Gemini API key configured in `app.py`. You can swap it to an environment variable for security.

