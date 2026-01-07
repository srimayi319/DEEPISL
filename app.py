import os
import sys
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
import uuid

# --- CONFIGURATION ---
N_FRAMES = 30
MIN_CONFIDENCE = 0.65
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FILE PATHS ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_final.tflite")
CLASS_NAMES_PATH = os.path.join(ROOT_DIR, "models", "label_encoder_final.npy")
GLOSS_MAP_PATH = os.path.join(ROOT_DIR, "gloss_map.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "static", "animations")

# --- FLASK SETUP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- GLOBAL VARIABLES ---
recognizer = None
generator = None
user_sessions = {}

# --- INITIALIZE MODELS ---
def initialize_models():
    global recognizer, generator
    
    # 1. Initialize Recognizer (Standard TF Version)
    try:
        print("="*50)
        print("Initializing ISL Recognizer (Standard TensorFlow)...")
        print(f"Model Path: {MODEL_PATH}")
        print(f"Labels Path: {CLASS_NAMES_PATH}")
        
        from isl_recognizer import ISLRecognizer
        recognizer = ISLRecognizer(MODEL_PATH, CLASS_NAMES_PATH)
        print("✅ ISL Recognizer initialized (Standard TensorFlow)")
    except ImportError as e:
        print(f"❌ ERROR loading Recognizer: {e}")
        print("Standard TensorFlow library might be missing.")
        print("Please run: pip install tensorflow")
        recognizer = None
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading Recognizer: {e}")
        recognizer = None
    
    # 2. Initialize Generator
    try:
        print("Initializing ISL Generator...")
        from isl_generator import ISLGenerator
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        generator = ISLGenerator(GLOSS_MAP_PATH, OUTPUT_DIR)
        print("✅ ISL Generator initialized")
        
    except Exception as e:
        print(f"⚠️  ERROR loading Generator: {e}")
        print("Recognition might work, but Text-to-ISL will fail.")
        generator = None

    # CRITICAL FIX: Stop server if recognizer fails
    if recognizer is None:
        print("!!! SERVER HALTED !!!")
        print("ISL Recognizer failed to load.")
        print("Please install 'tensorflow' to fix errors.")
        print("Server will NOT start.")
        sys.exit(1) # Force stop to avoid infinite errors

initialize_models()

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/api/predict_sequence", methods=["POST"])
def http_predict_sequence():
    if not recognizer:
        return jsonify({"error": "Recognizer not initialized (Standard TF). Install tensorflow."}), 503
    
    try:
        data = request.get_json()
        sequence = np.array(data.get("sequence", []), dtype=np.float32)
        history_of_signs = data.get("history", [])

        if sequence.shape != (N_FRAMES, 144):
            return jsonify({"error": f"Invalid sequence shape: {sequence.shape}"}), 400

        smoothed_label, confidence = recognizer.predict_sequence_smoothed(sequence)
        
        if confidence > MIN_CONFIDENCE:
            if not history_of_signs or history_of_signs[-1] != smoothed_label:
                history_of_signs.append(smoothed_label)
        
        sentence = " ".join(history_of_signs) if history_of_signs else ""
        return jsonify({
            "label": smoothed_label,
            "confidence": float(confidence),
            "sentence": sentence,
            "history": history_of_signs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate_animation", methods=["POST"])
def http_generate_animation():
    """HTTP endpoint for Text → ISL animation generation"""
    if generator is None:
        return jsonify({"error": "Animation generator not available"}), 500
    
    data = request.get_json()
    text = data.get("text", "").strip() if data else ""
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        video_path = generator.generate_video_from_text(text)
        if video_path and os.path.exists(video_path):
            # Create relative URL for the generated animation
            relative_url = os.path.relpath(video_path, start=ROOT_DIR)
            return jsonify({
                "video_url": f"/{relative_url.replace(os.sep, '/')}",
                "status": "success",
                "text": text
            })
        else:
            return jsonify({"error": "Could not generate animation"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- SOCKETIO EVENTS ---
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    user_sessions[client_id] = {
        'history': [], 
        'last_prediction_time': 0
    }
    print(f"Client connected: {client_id}")
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    user_sessions.pop(client_id, None)
    print(f"Client disconnected: {client_id}")

@socketio.on('predict_sequence')
def handle_prediction(data):
    client_id = request.sid
    
    if client_id not in user_sessions or not recognizer:
        emit('prediction_error', {'error': 'Session or model not available'})
        return
    
    try:
        sequence = np.array(data.get('sequence', []), dtype=np.float32)
        history = user_sessions[client_id]['history']

        if sequence.shape != (N_FRAMES, 144):
            emit('prediction_error', {'error': f'Invalid sequence shape: {sequence.shape}'})
            return
        
        smoothed_label, confidence = recognizer.predict_sequence_smoothed(sequence)
        
        print(f"📊 Prediction result: {smoothed_label} ({confidence:.3f})")
        
        if confidence > MIN_CONFIDENCE:
            if not history or history[-1] != smoothed_label:
                history.append(smoothed_label)
                if len(history) > 20:
                    history.pop(0)
        
        sentence = " ".join(history) if history else ""
        emit('prediction_result', {
            'label': smoothed_label,
            'confidence': float(confidence),
            'sentence': sentence,
            'history': history.copy()
        })
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        emit('prediction_error', {'error': str(e)})

@socketio.on('generate_animation')
def handle_generate_animation(data):
    """WebSocket: Text → ISL animation generation"""
    if generator is None:
        emit('animation_error', {'error': 'Animation generator not available'})
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
            emit('animation_result', {
                'video_url': video_url, 
                'text': text, 
                'status': 'success'
            })
            print(f"Animation generated: {video_url}")
        else:
            print(f"Error: Video generation failed or file not found.")
            emit('animation_error', {'error': 'Could not generate animation'})
            
    except Exception as e:
        print(f"Server Exception: {e}")
        emit('animation_error', {'error': str(e)})

@socketio.on('clear_history')
def handle_clear_history():
    client_id = request.sid
    if client_id in user_sessions:
        user_sessions[client_id]['history'] = []
        if recognizer: recognizer.clear_buffer()
        
    emit('prediction_result', {
        'label': '',
        'confidence': 0.0,
        'sentence': '',
        'history': []
    })

@socketio.on('clear_prediction_buffer')
def handle_clear_prediction_buffer():
    """Clears the smoothing buffer when transitioning to a new sign"""
    if recognizer:
        recognizer.clear_buffer()

if __name__ == "__main__":
    print("Starting ISL Recognition Server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)