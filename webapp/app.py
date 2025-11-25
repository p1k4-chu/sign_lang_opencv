import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
import mediapipe as mp
import base64
import os
import sys
import traceback

# --- IMPORT UTILS ---
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from hand_utils import extract_features, Smoother
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import hand_utils. {e}")
    sys.exit(1)

app = Flask(__name__)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hand_gesture_model.keras')
LABEL_PATH = os.path.join(BASE_DIR, 'model', 'class_names.npy')

# --- HARDCODED FALLBACK CLASSES ---
# Based on your notebook output: ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Y']
FALLBACK_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Y']

# --- LOAD AI MODEL ---
print(f"🔍 Loading AI from: {MODEL_PATH}")
model = None
class_names = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded!")
    else:
        print("❌ Model file missing!")

    if os.path.exists(LABEL_PATH):
        class_names = np.load(LABEL_PATH, allow_pickle=True) # allow_pickle needed for strings
        print(f"✅ Labels loaded from file: {class_names}")
    else:
        print("⚠️ Labels file missing! Using hardcoded fallback.")
        class_names = FALLBACK_CLASSES

except Exception as e:
    print(f"❌ Error initializing AI: {e}")
    # Ensure we have classes even if loading fails
    if class_names is None:
        class_names = FALLBACK_CLASSES

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
smoother = Smoother(alpha=0.85)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if model is None:
        return jsonify({'prediction': 'Error: No Model', 'confidence': 0.0})

    try:
        # 1. Decode Image
        data = request.json
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data'})

        if "," in image_data:
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
            
        binary_data = base64.b64decode(encoded)
        image_array = np.frombuffer(binary_data, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 2. Process Hand
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        prediction = "--"
        confidence = 0.0

        if results.multi_hand_landmarks:
            # Get primary hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Convert to numpy
            raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            
            # Smooth & Feature Extraction
            smoothed = smoother(raw_landmarks)
            features = extract_features(smoothed)
            
            # Reshape for Keras
            input_data = features.reshape(1, -1)
            
            # Predict
            probs = model.predict(input_data, verbose=0)[0]
            max_idx = np.argmax(probs)
            
            # --- LABEL LOOKUP FIX ---
            # Ensure we use the global class_names list
            current_classes = class_names if class_names is not None else FALLBACK_CLASSES
            
            if max_idx < len(current_classes):
                prediction = str(current_classes[max_idx])
            else:
                prediction = f"Unknown ({max_idx})"
                
            confidence = float(probs[max_idx])

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })

    except Exception as e:
        print("Processing Error:")
        traceback.print_exc()
        return jsonify({'prediction': 'ERR', 'error': str(e), 'confidence': 0.0})

if __name__ == '__main__':
    app.run(debug=True, port=5000)