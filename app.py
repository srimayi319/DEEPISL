import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
from collections import deque, Counter

# --- CONFIGURATION ---
N_FRAMES = 30
MIN_CONFIDENCE = 0.65
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FILE PATHS ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "realtime.tflite")
CLASS_NAMES_PATH = os.path.join(ROOT_DIR, "models", "label_encoder.npy")
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
    try:
        print("Initializing ISL Recognizer...")
        from isl_recognizer import ISLRecognizer
        recognizer = ISLRecognizer(MODEL_PATH, CLASS_NAMES_PATH)
        print("ISL Recognizer initialized")
        
        # Initialize animation generator if available
        try:
            from isl_generator import ISLGenerator
            generator = ISLGenerator(GLOSS_MAP_PATH, OUTPUT_DIR)
            print("ISL Generator initialized")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        except ImportError:
            print("ISL Generator not available - animation features disabled")
        except Exception as e:
            print(f"Error initializing generator: {e}")
            
    except Exception as e:
        print(f"Error initializing models: {e}")

initialize_models()

# --- UTILITY FUNCTIONS ---
def isl_to_english_sentence(history_of_signs):
    return " ".join(history_of_signs) if history_of_signs else ""

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

@app.route('/animations/<path:filename>')
def serve_animations(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    """Handle Chrome DevTools request"""
    return jsonify({"message": "Chrome DevTools endpoint"})

@app.route("/api/predict_sequence", methods=["POST"])
def http_predict_sequence():
    if not recognizer:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        data = request.get_json()
        
        # --- NEW: Catch conversion errors specifically ---
        try:
            sequence = np.array(data.get("sequence", []), dtype=np.float32)
        except ValueError:
            return jsonify({"error": "Invalid data format: Sequence must be numbers"}), 400
        # -------------------------------------------------

        history_of_signs = data.get("history", [])

        if sequence.shape != (N_FRAMES, 144):
            return jsonify({"error": f"Invalid sequence shape: {sequence.shape}"}), 400

        # Use smoothed prediction like OpenCV code
        smoothed_label, confidence = recognizer.predict_sequence_smoothed(sequence)
        
        # Apply confidence threshold
        if confidence > MIN_CONFIDENCE:
            if not history_of_signs or history_of_signs[-1] != smoothed_label:
                history_of_signs.append(smoothed_label)
        
        sentence = isl_to_english_sentence(history_of_signs)
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

@app.route("/api/status")
def status():
    return jsonify({
        "status": "running",
        "recognizer_loaded": recognizer is not None,
        "generator_loaded": generator is not None
    })

@app.route("/api/test")
def test_endpoint():
    return jsonify({"message": "Server running", "models_loaded": recognizer is not None})

# --- SOCKETIO EVENTS ---
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    user_sessions[client_id] = {
        'history': [], 
        'last_prediction_time': 0
    }
    emit('connection_response', {'status': 'connected'})
    print(f"Client connected: {client_id}")

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
        # Throttle predictions
        current_time = time.time()
        if current_time - user_sessions[client_id]['last_prediction_time'] < 0.3:
            return
        user_sessions[client_id]['last_prediction_time'] = current_time
        
        sequence = np.array(data.get('sequence', []), dtype=np.float32)
        history = user_sessions[client_id]['history']

        if sequence.shape != (N_FRAMES, 144):
            emit('prediction_error', {'error': f'Invalid sequence shape: {sequence.shape}'})
            return
        
        # Use smoothed prediction like OpenCV code
        smoothed_label, confidence = recognizer.predict_sequence_smoothed(sequence)
        
        # Apply confidence threshold
        if confidence > MIN_CONFIDENCE:
            if not history or history[-1] != smoothed_label:
                history.append(smoothed_label)
                if len(history) > 20:
                    history.pop(0)
        
        sentence = isl_to_english_sentence(history)
        emit('prediction_result', {
            'label': smoothed_label,
            'confidence': float(confidence),
            'sentence': sentence,
            'history': history.copy()
        })
    except Exception as e:
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
            emit('animation_error', {'error': 'Could not generate animation'})
    except Exception as e:
        emit('animation_error', {'error': str(e)})

@socketio.on('clear_history')
def handle_clear_history():
    client_id = request.sid
    if client_id in user_sessions:
        user_sessions[client_id]['history'] = []
        recognizer.clear_buffer()
        
    emit('prediction_result', {
        'label': '',
        'confidence': 0.0,
        'sentence': '',
        'history': []
    })

# --- MAIN ---
if __name__ == "__main__":
    print("Starting ISL Recognition Server...")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Classes: {len(np.load(CLASS_NAMES_PATH))} alphabets")
    print(f"Animation generator: {'Available' if generator else 'Not available'}")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)