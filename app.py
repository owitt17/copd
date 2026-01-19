from flask import Flask, render_template, request, jsonify, session
import os
import numpy as np
import tensorflow as tf
import librosa
import google.generativeai as genai

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "geminisecretkey"

# Load API key
GEMINI_API_KEY = "AIzaSyBWk-lOsDSmSjNOjgnW8_xKQsJ__jfgV-Q"
genai.configure(api_key=GEMINI_API_KEY)

# Define the system prompt with strict rules
SYSTEM_PROMPT = """
You are a chatbot that only answers questions about Chronic Obstructive Pulmonary Disease (COPD). 
You must strictly follow these rules:

1. **You must only talk about COPD.**  
   - If the user asks about anything else (e.g., cancer, diabetes, politics), say:  
     *'I can only answer questions about Chronic Obstructive Pulmonary Disease. Please ask me something related to COPD!'*

2. **Be very concise and structured.**  
   - Do NOT write long paragraphs.  
   - Always respond in a structured format like this:  

     **Example:**
     **Symptoms of COPD:**  
     - Shortness of breath  
     - Wheezing  
     - Chronic cough  

     **Treatments:**  
     - Medication  
     - Oxygen therapy  
     - Pulmonary rehabilitation  

3. **Be polite and friendly.**  
   - Always greet when the user greets you.  
   - Keep the response clear and easy to understand.

4. **Never make up information.**  
   - Stick to verified medical facts about COPD.  
"""

# Load the trained model
model = tf.keras.models.load_model('model_copd.h5')

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
        print(f"Error processing file {file_path}: {e}")
        return None

def classify_sound(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)
        prediction = model.predict(features)
        return "Healthy" if prediction[0][0] > 0.5 else "You have COPD"
    else:
        return "Error: Could not process the audio file."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/output')
def output():
    return render_template('output.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = classify_sound(file_path)
        os.remove(file_path)
        return jsonify({"result": result}), 200

    return jsonify({"error": "Invalid file format, only .wav allowed"}), 400

@app.route("/chatbot_api", methods=["POST"])
def chatbot_api():
    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Construct the full conversation
        prompt = SYSTEM_PROMPT + f"\n\nUser: {user_message}\nChatbot:"

        # Use Gemini API to generate response
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        bot_reply = response.text.strip()

        # If Gemini still answers non-COPD questions, filter response manually
        if "I can only answer questions about Chronic Obstructive Pulmonary Disease" not in bot_reply and "COPD" not in bot_reply:
            bot_reply = "I can only answer questions about Chronic Obstructive Pulmonary Disease. Please ask me something related to COPD!"

        # Ensure structured formatting
        formatted_reply = (
            bot_reply.replace("**", "<b>")  # Convert Markdown bold to HTML
            .replace("- ", "• ")  # Convert lists to bullet points
            .replace("\n\n", "<br><br>")  # Ensure paragraph spacing
        )

        return jsonify({"response": formatted_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
