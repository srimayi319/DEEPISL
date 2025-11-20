import tensorflow as tf
import numpy as np
from collections import deque, Counter

class ISLRecognizer:
    def __init__(self, model_path, class_names_path):
        """ISL Alphabet Recognizer - Simplified to match OpenCV version"""
        print("Loading ISL Recognizer...")
        
        # Load class names
        self.label_classes = np.load(class_names_path, allow_pickle=True)
        print(f"Loaded {len(self.label_classes)} classes")
        
        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Label buffer for smoothing (same as OpenCV code)
        self.label_buffer = deque(maxlen=5)
        
    def predict_sequence(self, sequence):
        try:
            if sequence.shape != (30, 144):
                return "error", 0.0
            
            # --- ADDED NORMALIZATION HERE ---
            # This MUST match exactly what you did in training
            sequence = np.array(sequence)
            mean = np.mean(sequence)
            std = np.std(sequence) + 1e-8
            sequence = (sequence - mean) / std
            # --------------------------------
            
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            pred_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][pred_idx])
            label = str(self.label_classes[pred_idx])
            
            return label, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0 
        
    def predict_sequence_smoothed(self, sequence):
        """
        Predict with label smoothing - same as OpenCV code
        """
        label, confidence = self.predict_sequence(sequence)
        
        # Apply confidence threshold and buffer logic
        if confidence > 0.65:
            self.label_buffer.append(label)
        
        # Return most common label from buffer (same as OpenCV)
        if len(self.label_buffer) > 0:
            smoothed_label = Counter(self.label_buffer).most_common(1)[0][0]
            return smoothed_label, confidence
        else:
            return label, confidence
    
    def clear_buffer(self):
        """Clear the label buffer"""
        self.label_buffer.clear()