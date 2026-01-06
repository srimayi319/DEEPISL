import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

# ================= CONFIG =================
MODEL_PATH = "models/model_final.tflite"
CLASS_NAMES_PATH = "models/label_encoder_final.npy"

N_FRAMES = 30
MIN_CONFIDENCE = 0.65
MOTION_THRESHOLD = 0.01   # adjust if needed

# ================= LOAD MODEL =================
print("Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
print("Classes:", class_names)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ================= KEYPOINT EXTRACTION =================
def extract_keypoints(hand_results, pose_results):
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for lm, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                left_hand = coords
            else:
                right_hand = coords

    pose = np.zeros(6 * 3)
    if pose_results.pose_landmarks:
        indices = [11, 12, 13, 14, 15, 16]
        pose = np.array([
            [
                pose_results.pose_landmarks.landmark[i].x,
                pose_results.pose_landmarks.landmark[i].y,
                pose_results.pose_landmarks.landmark[i].z
            ] for i in indices
        ]).flatten()

    return np.concatenate([left_hand, right_hand, pose])

# ================= PREDICTION =================
def predict(sequence):
    # EXACT SAME normalization as training
    sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)

    input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = np.argmax(output)
    return class_names[idx], float(output[idx])

# ================= MAIN LOOP =================
cap = cv2.VideoCapture(0)
sequence = []
prev_keypoints = None
motion_started = False
label_buffer = deque(maxlen=5)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands, mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    print("Press Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        pose_results = pose.process(rgb)

        keypoints = extract_keypoints(hand_results, pose_results)

        # ---------- MOTION DETECTION ----------
        motion = 0
        if prev_keypoints is not None:
            motion = np.linalg.norm(keypoints - prev_keypoints)
        prev_keypoints = keypoints

        if not motion_started:
            if motion > MOTION_THRESHOLD:
                motion_started = True
                sequence = []
                print("🟢 Motion detected — capturing sign")
            else:
                cv2.putText(frame, "WAITING FOR SIGN",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                cv2.imshow("ISL Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

        sequence.append(keypoints)

        # ---------- PREDICT ----------
        if len(sequence) == N_FRAMES:
            seq = np.array(sequence)
            label, conf = predict(seq)

            print(f"Prediction: {label} ({conf:.2f})")

            if conf > MIN_CONFIDENCE:
                label_buffer.append(label)
                final_label = Counter(label_buffer).most_common(1)[0][0]
            else:
                final_label = "..."

            sequence = []
            motion_started = False

        # ---------- DRAW ----------
        cv2.putText(frame,
                    f"Frames: {len(sequence)}/{N_FRAMES}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        cv2.imshow("ISL Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
