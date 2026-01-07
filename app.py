import os
import sys
import mimetypes
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit
import numpy as np

# ================== CONFIG ==================

N_FRAMES = 30
MIN_CONFIDENCE = 0.65

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_final.tflite")
CLASS_NAMES_PATH = os.path.join(ROOT_DIR, "models", "label_encoder_final.npy")
GLOSS_MAP_PATH = os.path.join(ROOT_DIR, "gloss_map.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "static", "animations")

mimetypes.add_type("video/mp4", ".mp4")

# ================== FLASK SETUP ==================

app = Flask(
    __name__,
    static_folder=None,
    template_folder=os.path.join(ROOT_DIR, "templates")
)

app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ================== GLOBALS ==================

recognizer = None
generator = None
user_sessions = {}

# ================== MODEL INIT ==================

def initialize_models():
    global recognizer, generator

    try:
        print("Initializing ISL Recognizer...")
        from isl_recognizer import ISLRecognizer
        recognizer = ISLRecognizer(MODEL_PATH, CLASS_NAMES_PATH)
        print("✅ Recognizer loaded")
    except Exception as e:
        print("❌ Recognizer failed:", e)
        sys.exit(1)

    try:
        print("Initializing ISL Generator...")
        from isl_generator import ISLGenerator
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        generator = ISLGenerator(GLOSS_MAP_PATH, OUTPUT_DIR)
        print("✅ Generator loaded")
    except Exception as e:
        print("⚠️ Generator failed:", e)
        generator = None

initialize_models()

# ================== ROUTES ==================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(ROOT_DIR, "js"), filename)

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(ROOT_DIR, "css"), filename)

# 🎯 CRITICAL FIX — REAL VIDEO STREAMING
@app.route("/animations/<path:filename>")
def serve_animation(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)

    print("🎬 VIDEO REQUEST:", file_path)
    print("📁 Exists:", os.path.exists(file_path))

    if not os.path.exists(file_path):
        return "Animation file not found", 404

    return send_file(
        file_path,
        mimetype="video/mp4",
        as_attachment=False,
        conditional=True  # ⭐ REQUIRED for browser video playback
    )

# ================== HTTP API ==================

@app.route("/api/predict_sequence", methods=["POST"])
def http_predict_sequence():
    if not recognizer:
        return jsonify({"error": "Recognizer not loaded"}), 503

    try:
        data = request.get_json()
        sequence = np.array(data.get("sequence", []), dtype=np.float32)
        history = data.get("history", [])

        if sequence.shape != (N_FRAMES, 144):
            return jsonify({"error": "Invalid shape"}), 400

        label, confidence = recognizer.predict_sequence_smoothed(sequence)

        if confidence > MIN_CONFIDENCE:
            if not history or history[-1] != label:
                history.append(label)

        return jsonify({
            "label": label,
            "confidence": float(confidence),
            "sentence": " ".join(history),
            "history": history
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate_animation", methods=["POST"])
def http_generate_animation():
    if not generator:
        return jsonify({"error": "Generator not loaded"}), 500

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    print("🧠 Generating animation:", text)
    path = generator.generate_video_from_text(text)

    if path and os.path.exists(path):
        filename = os.path.basename(path)
        return jsonify({
            "status": "success",
            "video_url": f"/animations/{filename}"
        })

    return jsonify({"error": "Generation failed"}), 500

# ================== SOCKET EVENTS ==================

@socketio.on("connect")
def handle_connect():
    user_sessions[request.sid] = {"history": []}
    print("Client connected:", request.sid)

@socketio.on("disconnect")
def handle_disconnect():
    user_sessions.pop(request.sid, None)
    print("Client disconnected:", request.sid)

@socketio.on("predict_sequence")
def handle_prediction(data):
    if not recognizer or request.sid not in user_sessions:
        emit("prediction_error", {"error": "Model/session error"})
        return

    try:
        seq = np.array(data.get("sequence", []), dtype=np.float32)
        history = user_sessions[request.sid]["history"]

        if seq.shape != (N_FRAMES, 144):
            emit("prediction_error", {"error": "Invalid shape"})
            return

        label, confidence = recognizer.predict_sequence_smoothed(seq)

        if confidence > MIN_CONFIDENCE:
            if not history or history[-1] != label:
                history.append(label)
                history[:] = history[-20:]

        emit("prediction_result", {
            "label": label,
            "confidence": float(confidence),
            "sentence": " ".join(history),
            "history": history
        })

    except Exception as e:
        emit("prediction_error", {"error": str(e)})

@socketio.on("generate_animation")
def handle_animation(data):
    if not generator:
        emit("animation_error", {"error": "Generator not loaded"})
        return

    text = data.get("text", "").strip()
    print("🎬 Socket generation:", text)

    path = generator.generate_video_from_text(text)

    if path and os.path.exists(path):
        filename = os.path.basename(path)
        emit("animation_result", {
            "status": "success",
            "video_url": f"/animations/{filename}"
        })
    else:
        emit("animation_error", {"error": "Generation failed"})

@socketio.on("clear_history")
def clear_history():
    if request.sid in user_sessions:
        user_sessions[request.sid]["history"] = []
    if recognizer:
        recognizer.clear_buffer()

# ================== START ==================

if __name__ == "__main__":
    print("🚀 Starting DeepISL server...")
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
