import cv2
import numpy as np
import os
import time
import mediapipe as mp

# ================= CONFIG =================
BASE_DIR = "isl_dataset_clean"
LABELS = ["man"]
REM=["hello","bye","Thankyou","sorry","what","how","name","im fine","indian","sign","language","me","you","hearing","deaf","woman"]
SAMPLES_PER_LABEL = 10
N_FRAMES = 30
COUNTDOWN_SECONDS = 3

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ================= KEYPOINT EXTRACTION =================
def extract_keypoints(hand_results, pose_results):
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            label = handedness.classification[0].label
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()

            if label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

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

# ================= COLLECTION =================
def collect():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for label in LABELS:
            os.makedirs(f"{BASE_DIR}/{label}", exist_ok=True)

            # ---------- WAIT FOR START ONCE PER LABEL ----------
            print(f"\nREADY FOR LABEL: {label}")
            print("Press 'S' to START this label")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                cv2.putText(frame, f"Label: {label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                cv2.putText(frame, "Press 'S' to START", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                cv2.imshow("ISL Data Collection", frame)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('s'):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # ---------- SAMPLES ----------
            for sample in range(SAMPLES_PER_LABEL):
                print(f"Recording {label} | Sample {sample+1}/{SAMPLES_PER_LABEL}")
                sequence = []

                while len(sequence) < N_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(rgb)
                    pose_results = pose.process(rgb)

                    keypoints = extract_keypoints(hand_results, pose_results)
                    sequence.append(keypoints)

                    # ---- DRAW LANDMARKS ----
                    if hand_results.multi_hand_landmarks:
                        for hand_lms in hand_results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    if pose_results.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    cv2.putText(frame,
                                f"{label} | {sample+1}/{SAMPLES_PER_LABEL} | Frame {len(sequence)}/{N_FRAMES}",
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0,255,0), 2)

                    cv2.imshow("ISL Data Collection", frame)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # ---- SAVE ----
                np.save(f"{BASE_DIR}/{label}/{sample+10}.npy", np.array(sequence))
                print("Saved ✔")

                # ---------- COUNTDOWN ----------
                for sec in range(COUNTDOWN_SECONDS, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    cv2.putText(frame,
                                f"Next sample in {sec}...",
                                (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0,0,255),
                                3)
                    cv2.imshow("ISL Data Collection", frame)
                    cv2.waitKey(1000)

    cap.release()
    cv2.destroyAllWindows()

collect()
