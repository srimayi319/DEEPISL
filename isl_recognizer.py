import tensorflow as tf
import numpy as np
import os
import time
from functools import lru_cache

class ISLRecognizer:
    def __init__(self, model_path, labels_path, norm_stats_path):
        """Optimized ISL Recognizer with caching and performance improvements"""
        print("Loading TFLite model...")
        
        # Use multiple threads for better performance
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=4
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.label_classes = np.load(labels_path, allow_pickle=True)
        
        print(f"Model loaded. Input shape: {self.input_details[0]['shape']}")
        print(f"Number of classes: {len(self.label_classes)}")
        
        # Load normalization stats
        self.mean_vector = None
        self.std_vector = None
        self._load_normalization_stats(norm_stats_path)
        
        # Pre-compute for faster normalization
        if self.mean_vector is not None and self.std_vector is not None:
            self.inv_std_vector = 1.0 / (self.std_vector + 1e-8)
            print("Normalization stats loaded and pre-computed")
        else:
            print("Warning: Using unnormalized data")
            self.inv_std_vector = None
        
        # Warm up the model
        self._warm_up()
        
        # Prediction cache
        self._prediction_cache = {}
        self._cache_size = 50
        
    def _load_normalization_stats(self, path):
        """Load normalization statistics"""
        if os.path.exists(path):
            try:
                norm_stats = np.load(path, allow_pickle=True).item()
                self.mean_vector = norm_stats['mean'].astype(np.float32)
                self.std_vector = norm_stats['std'].astype(np.float32)
                print(f"Normalization stats loaded: mean={self.mean_vector.shape}, std={self.std_vector.shape}")
            except Exception as e:
                print(f"Warning: Failed to load normalization stats: {e}")
    
    def _warm_up(self):
        """Warm up the model with dummy data"""
        try:
            dummy_input = np.random.randn(30, 144).astype(np.float32)
            self.predict_sequence(dummy_input)
            print("Model warmed up successfully")
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")
    
    def _get_sequence_hash(self, sequence):
        """Generate hash for caching"""
        return hash(sequence.tobytes())
    
    def predict_sequence(self, sequence):
        """
        Optimized prediction with caching and performance improvements
        sequence shape must be (30, 144)
        """
        try:
            # Validate input shape
            if sequence.shape != (30, 144):
                raise ValueError(f"Expected shape (30, 144), got {sequence.shape}")
            
            # Check cache
            seq_hash = self._get_sequence_hash(sequence)
            if seq_hash in self._prediction_cache:
                return self._prediction_cache[seq_hash]
            
            # Convert to contiguous array for better performance
            input_sequence = np.ascontiguousarray(sequence, dtype=np.float32)
            
            # Apply normalization if stats are available
            if self.mean_vector is not None and self.inv_std_vector is not None:
                # Vectorized normalization (much faster)
                input_sequence = (input_sequence - self.mean_vector) * self.inv_std_vector
            
            # Prepare input tensor (shape: [1, 30, 144])
            input_data = np.expand_dims(input_sequence, axis=0).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process results efficiently
            idx = np.argmax(output_data[0])  # Direct indexing is faster
            confidence = float(output_data[0][idx])
            
            label = self.label_classes[idx] if idx < len(self.label_classes) else "unknown"
            
            # Cache high-confidence predictions
            if confidence > 0.6:  # Only cache confident predictions
                if len(self._prediction_cache) >= self._cache_size:
                    # Remove oldest entry
                    self._prediction_cache.pop(next(iter(self._prediction_cache)))
                self._prediction_cache[seq_hash] = (label, confidence)
            
            if inference_time > 0.1:  # Log slow predictions
                print(f"Slow prediction: {inference_time:.3f}s - {label} ({confidence:.2f})")
                
            return label, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0
    
    def batch_predict(self, sequences):
        """Predict multiple sequences (for future use)"""
        results = []
        for sequence in sequences:
            results.append(self.predict_sequence(sequence))
        return results