import os
import json
import cv2
import numpy as np
from uuid import uuid4
from collections import defaultdict
import mediapipe as mp
import spacy
import re

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print("spaCy model loaded successfully")
except OSError:
    print("Downloading spaCy English model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print("spaCy model loaded successfully")

mp_face_mesh = mp.solutions.face_mesh

# Constants
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
SKELETON_COLOR = (171, 134, 46)
HAND_COLOR = (111, 107, 229)
JOINT_COLOR = (79, 199, 249)
FACE_COLOR = (224, 224, 224)
FPS_ANIM = 25
IMG_SIZE = (512, 512)

class ISLGenerator:
    def __init__(self, gloss_map_path, data_dir):
        self.gloss_map_path = gloss_map_path
        self.data_dir = data_dir
        self.img_size = IMG_SIZE
        self.fps = FPS_ANIM
        
        self.auxiliary_verbs = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'a',
                                 'do', 'does', 'did', 'have', 'has', 'had', 'will', 'shall'}
        
        self.question_words = {'what', 'where', 'when', 'why', 'how', 'which', 'who', 'whom'}
        
        self.pronoun_map = {
            'i': 'me', 'my': 'me', 'mine': 'me', 'im':'me',
            'you': 'you', 'your': 'you', 'yours': 'you',
            'he': 'he', 'him': 'he', 'his': 'he',
            'she': 'she', 'her': 'she', 'hers': 'she',
            'we': 'we', 'us': 'we', 'our': 'we',
            'they': 'they', 'them': 'they', 'their': 'they',
            'it': 'it', 'its': 'it'
        }
        
        self.gloss_map, self.phrase_trie = self._load_gloss_map()

    def _load_gloss_map(self):
        """Load gloss_map.json and build a phrase trie."""
        gloss_map = {}
        phrase_trie = {}
        try:
            with open(self.gloss_map_path, 'r') as f:
                raw_map = json.load(f)
                gloss_map = {k.lower(): v for k, v in raw_map.items()}

            # Resolve paths relative to script directory
            for key, path in gloss_map.items():
                if not os.path.isabs(path):
                    gloss_map_dir = os.path.dirname(os.path.abspath(self.gloss_map_path))
                    full_path = os.path.join(gloss_map_dir, path)
                    gloss_map[key] = full_path

            for phrase in gloss_map.keys():
                words = phrase.split()
                node = phrase_trie
                for word in words:
                    if word not in node:
                        node[word] = {}
                    node = node[word]
                node['__END__'] = phrase
        except FileNotFoundError:
            print(f"Warning: {self.gloss_map_path} not found. Text-to-ISL will not work.")
        return gloss_map, phrase_trie

    def _is_color_word(self, word):
        colors = {'red', 'blue', 'green', 'yellow', 'black', 'white', 
                  'orange', 'pink', 'purple', 'brown', 'gray', 'grey',
                  'violet', 'indigo', 'magenta', 'cyan', 'turquoise'}
        return word.lower() in colors

    def _preprocess_contractions(self, text):
        """Replaces common contractions so spaCy doesn't split them weirdly."""
        text = text.lower()
        replacements = {
            "i'm": "i am", "im": "i am", "you're": "you are",
            "we're": "we are", "they're": "they are", "he's": "he is",
            "she's": "she is", "it's": "it is"
        }
        for contraction, replacement in replacements.items():
            pattern = re.compile(r"\b" + re.escape(contraction) + r"\b")
            text = pattern.sub(" " + replacement + " ", text)
        return text.strip()

    def _spacy_pos_tagging(self, sentence):
        """Use spaCy for POS tagging."""
        doc = nlp(sentence.lower())
        tokens, tags = [], []
        for token in doc:
            if not token.is_alpha: continue
            tokens.append(token.text)
            tags.append(token.pos_)
        return tokens, tags

    def apply_isl_grammar(self, sentence):
        """Apply ISL grammar rules."""
        if not sentence or not sentence.strip(): return []
        
        tokens, tags = self._spacy_pos_tagging(sentence)
        subjects, objects, verbs, adjectives, colors, negations, questions, others = [], [], [], [], [], [], [], []
        
        for i, (word, pos_tag) in enumerate(zip(tokens, tags)):
            if word in self.auxiliary_verbs: continue
            if word in self.question_words: questions.append(word)
            elif word in ['not', 'no', 'never', 'nothing']: negations.append(word)
            elif self._is_color_word(word): colors.append(word)
            elif pos_tag == 'PRON' or (pos_tag == 'NOUN' and not subjects and not objects):
                subjects.append(self.pronoun_map.get(word, word))
            elif pos_tag == 'NOUN': objects.append(word)
            elif pos_tag == 'VERB': verbs.append(word)
            elif pos_tag == 'ADJ': adjectives.append(word)
            else: others.append(word)

        # ISL Order: Subject + Object + Verb + Adjectives + Others + Colors + Negation + Questions
        return subjects + objects + verbs + adjectives + others + colors + negations + questions

    def _text_to_gloss_sequence(self, tokens):
        if not tokens: return []
        glosses = []
        i = 0
        while i < len(tokens):
            current_node = self.phrase_trie
            longest_match_len = 0
            longest_match_gloss = None
            for j in range(i, len(tokens)):
                word = tokens[j]
                if word in current_node:
                    current_node = current_node[word]
                    if '__END__' in current_node:
                        longest_match_len = j - i + 1
                        longest_match_gloss = current_node['__END__']
                else:
                    break
            if longest_match_gloss:
                glosses.append(longest_match_gloss)
                i += longest_match_len
            else:
                word = tokens[i]
                if word in self.gloss_map:
                    glosses.append(word)
                else:
                    for char in word:
                        if char in self.gloss_map:
                            glosses.append(char)
                        else:
                            print(f"Warning: No gloss for character '{char}', skipping.")
                i += 1
        return glosses

    def text_to_gloss(self, sentence):
        sentence = self._preprocess_contractions(sentence)
        common_phrases = {
            'good morning': 2, 'good afternoon': 2, 
            'good evening': 2, 'good night': 2,
            'thank you': 2, 'hello': 1
        }
        initial_glosses = []
        remaining_sentence = sentence
        for phrase, length in common_phrases.items():
            if sentence.startswith(phrase):
                if phrase in self.gloss_map:
                    initial_glosses.append(phrase) 
                    remaining_sentence = sentence[len(phrase):].strip()
                    break
        if remaining_sentence:
            remaining_tokens = [token.text for token in nlp(remaining_sentence) if token.is_alpha]
            processed_remaining_tokens = self.apply_isl_grammar(" ".join(remaining_tokens))
            remaining_glosses = self._text_to_gloss_sequence(processed_remaining_tokens)
        else:
            remaining_glosses = []
        return initial_glosses + remaining_glosses

    def _draw_skeleton_on_frame(self, canvas, frame_data):
        if not frame_data: return
        def get_point_coords(point):
            return (int(point['x'] * self.img_size[0]), int(point['y'] * self.img_size[1]))
        def fill_torso(pose_points):
            if not pose_points or len(pose_points) < 25: return
            required_indices = [11, 12, 23, 24]
            if all(idx < len(pose_points) and pose_points[idx] for idx in required_indices):
                pts = [pose_points[11], pose_points[12], pose_points[24], pose_points[23]]
                pts_array = np.array([[int(p['x']*self.img_size[0]), int(p['y']*self.img_size[1])] for p in pts], np.int32)
                cv2.fillPoly(canvas, [pts_array], (200, 150, 100))
        def draw_smiling_face(face_points):
            if not face_points:
                center_x, center_y = self.img_size[0] // 2, self.img_size[1] // 3
                head_radius = 40
                cv2.circle(canvas, (center_x, center_y), head_radius, (255, 224, 189), -1)
                cv2.circle(canvas, (center_x, center_y), head_radius, (0, 0, 0), 2)
                eye_y = center_y - 10
                eye_offset = 15
                cv2.circle(canvas, (center_x - eye_offset, eye_y), 5, (0, 0, 0), -1)
                cv2.circle(canvas, (center_x + eye_offset, eye_y), 5, (0, 0, 0), -1)
                mouth_y = center_y + 10
                cv2.ellipse(canvas, (center_x, mouth_y), (20, 15), 0, 0, 180, (0, 0, 0), 3)
                return
            xs = [p['x'] for p in face_points if 'x' in p]
            ys = [p['y'] for p in face_points if 'y' in p]
            if not xs or not ys: return
            center_x = int(np.mean(xs) * self.img_size[0])
            center_y = int(np.mean(ys) * self.img_size[1])
            head_radius = 40
            cv2.circle(canvas, (center_x, center_y), head_radius, (255, 224, 189), -1)
            cv2.circle(canvas, (center_x, center_y), head_radius, (0, 0, 0), 2)
            eye_y = center_y - 10
            eye_offset = 15
            cv2.circle(canvas, (center_x - eye_offset, eye_y), 5, (0, 0, 0), -1)
            cv2.circle(canvas, (center_x + eye_offset, eye_y), 5, (0, 0, 0), -1)
            mouth_y = center_y + 10
            cv2.ellipse(canvas, (center_x, mouth_y), (20, 15), 0, 0, 180, (0, 0, 0), 3)
        def draw_connections(points, connections, color, thickness=3):
            if not points: return
            for start_idx, end_idx in connections:
                if (len(points) > max(start_idx, end_idx) and 
                    points[start_idx] and points[end_idx] and
                    'x' in points[start_idx] and 'y' in points[start_idx] and
                    'x' in points[end_idx] and 'y' in points[end_idx]):
                    cv2.line(canvas, get_point_coords(points[start_idx]),
                             get_point_coords(points[end_idx]), color, thickness, cv2.LINE_AA)
        def draw_points(points, color, size=4):
            if not points: return
            for point in points:
                if point and 'x' in point and 'y' in point:
                    cv2.circle(canvas, get_point_coords(point), size, color, -1, cv2.LINE_AA)
        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
        ]
        POSE_CONNECTIONS = [(11,12),(12,14),(14,16),(11,13),(13,15),(12,24),(11,23),(23,24)]
        pose_data = frame_data.get("pose", [])
        left_hand_data = frame_data.get("left_hand", [])
        right_hand_data = frame_data.get("right_hand", [])
        fill_torso(pose_data)
        draw_smiling_face(frame_data.get("face", []))
        draw_connections(pose_data, POSE_CONNECTIONS, SKELETON_COLOR, thickness=3)
        draw_connections(left_hand_data, HAND_CONNECTIONS, HAND_COLOR, thickness=2)
        draw_connections(right_hand_data, HAND_CONNECTIONS, HAND_COLOR, thickness=2)

    def _draw_label(self, canvas, text):
        if not text: return
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        color = TEXT_COLOR
        org = (30, 50)
        cv2.putText(canvas, text.upper(), org, font, font_scale, color, thickness, cv2.LINE_AA)
    
    def generate_video_from_text(self, text: str) -> str:
        try:
            tokens = self.text_to_gloss(text)
            if not tokens:
                print("No tokens generated from text")
                return None
            video_segments = []
            for token in tokens:
                json_path = self.gloss_map.get(token.lower(), None) 
                if json_path and os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            sign_data = json.load(f)
                            video_segments.append((token.upper(), sign_data))
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON in {json_path}")
                else:
                    print(f"Warning: No data for token: '{token}'")
            if not video_segments:
                print("No pose data collected")
                return None
            os.makedirs(self.data_dir, exist_ok=True)
            final_name = f"animation_{uuid4().hex[:8]}.mp4"
            final_path = os.path.join(self.data_dir, final_name)

            # Encoder setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out = cv2.VideoWriter(final_path, fourcc, self.fps, self.img_size)
            if not video_out.isOpened():
                print("mp4v failed, trying XVID...")
                final_path = final_path.replace(".mp4", ".avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_out = cv2.VideoWriter(final_path, fourcc, self.fps, self.img_size)

            if not video_out.isOpened():
                raise RuntimeError("❌ No supported video encoder found on this system.")

            for label, frames in video_segments:
                for frame_data in frames:
                    canvas = np.full((self.img_size[1], self.img_size[0], 3), 255, dtype=np.uint8)
                    canvas[:] = BG_COLOR
                    self._draw_skeleton_on_frame(canvas, frame_data)
                    self._draw_label(canvas, label)
                    video_out.write(canvas)

            video_out.release()
            print(f"Video saved at: {final_path}")

            # --- CRITICAL RENDER FIX ---
            # Force the file to be readable by the web server (Permissions 644)
            try:
                os.chmod(final_path, 0o644)
                print(f"✅ Permissions set for {final_name}")
            except Exception as e:
                print(f"⚠️ Could not set permissions: {e}")
            # -------------------------

            return final_path

        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return None