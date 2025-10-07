import os
import json
import time
from flask import Flask, render_template, request, jsonify,send_from_directory  
from flask_socketio import SocketIO, emit
from isl_recognizer import ISLRecognizer
from isl_generator import ISLGenerator
from utils import isl_to_english_sentence
import numpy as np

# --- CONFIGURATION ---
N_FRAMES = 30
MIN_CONFIDENCE = 0.50
FPS_ANIM = 25

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "real (1).tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "label_encoder2 (1).npy")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.npy")
GLOSS_MAP_PATH = os.path.join(ROOT_DIR, "gloss_map.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "static", "animations")

# --- FLASK & SOCKETIO SETUP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- GLOBAL VARIABLES ---
recognizer = None
generator = None
user_sessions = {}  # Track client sessions

# --- INITIALIZE MODELS ---
def initialize_models():
    global recognizer, generator
    try:
        print("Initializing models...")
        recognizer = ISLRecognizer(MODEL_PATH, LABELS_PATH, NORM_STATS_PATH)
        generator = ISLGenerator(GLOSS_MAP_PATH, OUTPUT_DIR)
        print("Models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")

initialize_models()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")
@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)
@app.route("/api/predict_sequence", methods=["POST"])
def http_predict_sequence():
    """HTTP fallback for ISL → Text prediction"""
    if recognizer is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        keypoint_sequence = np.array(data.get("sequence", []), dtype=np.float32)
        history_of_signs = data.get("history", [])

        if keypoint_sequence.shape != (N_FRAMES, 144):
            return jsonify({"error": f"Invalid sequence shape: {keypoint_sequence.shape}"}), 400

        label, confidence = recognizer.predict_sequence(keypoint_sequence)
        if confidence > MIN_CONFIDENCE:
            if not history_of_signs or history_of_signs[-1] != label:
                history_of_signs.append(label)
        
        sentence = isl_to_english_sentence(history_of_signs)
        return jsonify({
            "label": label,
            "confidence": float(confidence),
            "sentence": sentence,
            "history": history_of_signs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate_animation", methods=["POST"])
def http_generate_animation():
    """HTTP fallback for Text → ISL animation"""
    if generator is None:
        return jsonify({"error": "Generator not initialized"}), 500
    
    data = request.get_json()
    text = data.get("text", "").strip() if data else ""
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        video_path = generator.generate_video_from_text(text)
        if video_path and os.path.exists(video_path):
            relative_url = os.path.relpath(video_path, start=ROOT_DIR)
            return jsonify({"video_url": f"/{relative_url.replace(os.sep, '/')}"})
        else:
            return jsonify({"error": "Could not generate animation"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/test", methods=["GET"])
def test_endpoint():
    return jsonify({"message": "Server running", "models_loaded": recognizer is not None})

# --- SOCKETIO EVENTS ---
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    user_sessions[client_id] = {'history': [], 'last_prediction_time': 0}
    emit('connection_response', {'status': 'connected'})
    print(f"Client connected: {client_id}")

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    user_sessions.pop(client_id, None)
    print(f"Client disconnected: {client_id}")

@socketio.on('predict_sequence')
def handle_prediction(data):
    """WebSocket: ISL → Text real-time prediction"""
    client_id = request.sid
    if client_id not in user_sessions:
        emit('prediction_error', {'error': 'Session not found'})
        return
    if recognizer is None:
        emit('prediction_error', {'error': 'Model not initialized'})
        return
    
    try:
        current_time = time.time()
        if current_time - user_sessions[client_id]['last_prediction_time'] < 0.3:
            return  # throttle 3/sec
        user_sessions[client_id]['last_prediction_time'] = current_time
        
        keypoint_sequence = np.array(data.get('sequence', []), dtype=np.float32)
        history_of_signs = user_sessions[client_id]['history']

        if keypoint_sequence.shape != (N_FRAMES, 144):
            emit('prediction_error', {'error': f'Invalid sequence shape: {keypoint_sequence.shape}'})
            return
        
        label, confidence = recognizer.predict_sequence(keypoint_sequence)
        if confidence > MIN_CONFIDENCE:
            if not history_of_signs or history_of_signs[-1] != label:
                history_of_signs.append(label)
                if len(history_of_signs) > 20:
                    history_of_signs.pop(0)
        
        final_sentence = isl_to_english_sentence(history_of_signs)
        emit('prediction_result', {
            'label': label,
            'confidence': float(confidence),
            'sentence': final_sentence,
            'history': history_of_signs.copy()
        })
    except Exception as e:
        emit('prediction_error', {'error': str(e)})

@socketio.on('generate_animation')
def handle_generate_animation(data):
    """WebSocket: Text → ISL animation"""
    if generator is None:
        emit('animation_error', {'error': 'Generator not initialized'})
        return
    
    try:
        text = data.get('text', '').strip()
        if not text:
            emit('animation_error', {'error': 'No text provided'})
            return
        
        print(f"Generating animation for text: {text}")
        video_path = generator.generate_video_from_text(text)
        
        if video_path and os.path.exists(video_path):
            relative_url = os.path.relpath(video_path, start=ROOT_DIR)
            video_url = f"/{relative_url.replace(os.sep, '/')}"
            emit('animation_result', {'video_url': video_url, 'text': text, 'status': 'success'})
            print(f"Animation generated: {video_url}")
        else:
            emit('animation_error', {'error': 'Could not generate animation'})
    except Exception as e:
        emit('animation_error', {'error': str(e)})

@socketio.on('clear_history')
def handle_clear_history():
    """WebSocket event to clear server-side history for a client."""
    client_id = request.sid
    if client_id in user_sessions:
        user_sessions[client_id]['history'] = []
    emit('prediction_result', {
        'label': 'idle',
        'confidence': 1.0,
        'sentence': '',
        'history': []
    })
    print(f"History cleared for client: {client_id}")

# --- MAIN ---
if __name__ == "__main__":
    print("Starting Flask + SocketIO server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)