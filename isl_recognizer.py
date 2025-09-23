import tensorflow as tf
import numpy as np

class ISLRecognizer:
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.label_classes = np.load(labels_path, allow_pickle=True)
    
    def predict_sequence(self, sequence):
        """Performs inference on a sequence of keypoints."""
        # Reshape for model input: (1, N_FRAMES, FEATURE_DIM)
        input_data = np.asarray([sequence], dtype=np.float32)
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            idx = np.argmax(output_data, axis=-1)[0]
            confidence = float(output_data[0, idx]) 
            
            if idx < len(self.label_classes):
                label = self.label_classes[idx]
            else:
                label = "unknown"
                
            return label, confidence
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "error", 0.0