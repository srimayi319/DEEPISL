import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from isl_recognizer import ISLRecognizer
from isl_generator import ISLGenerator
from utils import isl_to_english_sentence
import numpy as np

# --- CONFIGURATION ---
N_FRAMES = 30
MIN_CONFIDENCE = 0.50
STRICT_CONFIDENCE = 0.92
STRICT_SIGNS = ["he", "she"]
FPS_ANIM = 25

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_trans.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "label_encoder.npy")
GLOSS_MAP_PATH = os.path.join(ROOT_DIR, "gloss_map.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "static", "animations")

app = Flask(__name__)

try:
    recognizer = ISLRecognizer(MODEL_PATH, LABELS_PATH)
    generator = ISLGenerator(GLOSS_MAP_PATH, OUTPUT_DIR)
except Exception as e:
    print(f"Error initializing models: {e}")
    recognizer = None
    generator = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict_sequence", methods=["POST"])
def predict_sequence():
    """
    Receives a sequence of keypoints, a history of recognized signs,
    and returns the full, grammatically corrected sentence.
    """
    print("=== PREDICT_SEQUENCE ENDPOINT CALLED ===")
    
    if recognizer is None:
        print("ERROR: Model not initialized")
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
    
    # Check for required keys
    if "sequence" not in data or "history" not in data:
        print(f"ERROR: Missing 'sequence' or 'history' key. Received keys: {list(data.keys())}")
        return jsonify({"error": "Missing 'sequence' or 'history' key in request"}), 400
    
    # Process data
    try:
        keypoint_sequence = np.array(data["sequence"], dtype=np.float32)
        history_of_signs = data["history"]
        print(f"Received keypoint sequence shape: {keypoint_sequence.shape}")
        print(f"Received history: {history_of_signs}")
    except Exception as e:
        print(f"ERROR converting data to numpy array: {e}")
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
    
    try:
        label, confidence = recognizer.predict_sequence(keypoint_sequence)
        print(f"Prediction result - Label: {label}, Confidence: {confidence}")
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Append the new label to the history if it's confident
    if confidence > MIN_CONFIDENCE:
        history_of_signs.append(label)
        print(f"Updated history: {history_of_signs}")
    
    # Process the raw signs with NLP function
    try:
        final_sentence = isl_to_english_sentence(history_of_signs)
        print(f"Final sentence: {final_sentence}")
    except Exception as e:
        print(f"ERROR in NLP processing: {e}")
        final_sentence = " ".join(history_of_signs)
    
    response = {
        "label": label,
        "confidence": float(confidence),
        "sentence": final_sentence,
        "history": history_of_signs
    }
    
    print(f"Returning response: {response}")
    print("=== REQUEST PROCESSING COMPLETE ===\n")
    
    return jsonify(response)

@app.route("/api/generate_animation", methods=["POST"])
def generate_animation():
    if generator is None:
        return jsonify({"error": "Generator not initialized"}), 500
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Invalid request payload"}), 400
    text = data["text"]
    try:
        video_path = generator.generate_video_from_text(text)
        if video_path:
            relative_url = os.path.relpath(video_path, start=ROOT_DIR)
            return jsonify({
                "video_url": f"/{relative_url}"
            })
        else:
            return jsonify({"error": "Could not generate animation. Missing data for some words."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a test endpoint to verify the backend is working
@app.route("/api/test", methods=["GET", "POST"])
def test_endpoint():
    if request.method == "POST":
        data = request.get_json() or {}
        return jsonify({
            "message": "Test endpoint working",
            "received_data": data,
            "status": "success"
        })
    
    return jsonify({
        "message": "Test GET endpoint working",
        "status": "success"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)