class MediaPipeManager {
    constructor() {
        this.holistic = null;
        this.capturing = false;
        this.frameCounter = 0;
        this.keypointSequence = [];
        this.lastPredictionTime = 0;
        this.video = null;
        this.canvas = null;
        this.canvasCtx = null;
        this.onPredictionReady = null;
        this.onError = null;
        
        // Debug counters
        this.framesProcessed = 0;
        this.predictionsAttempted = 0;
        this.landmarksDetected = {
            leftHand: 0,
            rightHand: 0,
            pose: 0
        };
    }

    async initialize() {
        try {
            console.log('🔄 Initializing MediaPipe Manager...');
            
            this.video = document.createElement('video');
            this.canvas = document.getElementById(ELEMENTS.LIVE_CANVAS);
            
            if (!this.canvas) {
                throw new Error('Canvas element not found');
            }
            
            this.canvasCtx = this.canvas.getContext('2d');

            // Initialize MediaPipe Holistic
            this.holistic = new Holistic({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
            });

            // Set options matching your OpenCV configuration
            this.holistic.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                refineFaceLandmarks: false,
                minDetectionConfidence: 0.5,  // Match your OpenCV: 0.5
                minTrackingConfidence: 0.5    // Match your OpenCV: 0.5
            });

            this.holistic.onResults((results) => this.onResults(results));
            this.holistic.onError((error) => this.handleError(error));
            
            console.log('✅ MediaPipe Holistic initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize MediaPipe:', error);
            this.handleError('MEDIAPIPE_INIT_ERROR');
            return false;
        }
    }

    async startCapture() {
        if (this.capturing) {
            console.log('⚠️ Already capturing video');
            return false;
        }

        try {
            console.log('📷 Requesting camera access...');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    frameRate: { ideal: 30 }
                } 
            });
            
            this.video.srcObject = stream;
            this.capturing = true;
            this.keypointSequence = [];
            this.frameCounter = 0;
            this.framesProcessed = 0;
            this.predictionsAttempted = 0;
            
            console.log('✅ Camera access granted, starting video stream...');
            
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    console.log(`🎥 Video stream: ${this.video.videoWidth}x${this.video.videoHeight}`);
                    
                    this.video.play();
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    
                    // Start processing frames
                    this.processVideoFrame();
                    resolve(true);
                };
                
                this.video.onerror = (error) => {
                    console.error('❌ Video playback error:', error);
                    this.handleError('VIDEO_PLAYBACK_ERROR');
                    resolve(false);
                };
            });

        } catch (err) {
            console.error("❌ Error accessing webcam: ", err);
            this.handleError('CAMERA_ACCESS_ERROR');
            return false;
        }
    }

    stopCapture() {
        if (this.capturing) {
            console.log('🛑 Stopping video capture...');
            this.capturing = false;
            
            if (this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(track => {
                    track.stop();
                    console.log('📹 Track stopped:', track.kind);
                });
                this.video.srcObject = null;
            }
            
            this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            console.log('✅ Video capture stopped');
        }
    }

    async processVideoFrame() {
        if (!this.capturing) return;
        
        this.frameCounter++;
        
        // Process every frame (remove frame skipping for better accuracy)
        try {
            await this.holistic.send({ image: this.video });
        } catch (error) {
            console.error('❌ Error processing frame:', error);
        }
        
        requestAnimationFrame(() => this.processVideoFrame());
    }

    onResults(results) {
        this.framesProcessed++;
        
        // Draw landmarks on canvas
        this.drawResults(results);
        
        // Extract keypoints and store in sequence
        const keypoints = this.extractKeypoints(results);
        this.keypointSequence.push(keypoints);
        
        // Log sequence progress periodically
        if (this.framesProcessed % 30 === 0) {
            console.log(`📊 Sequence progress: ${this.keypointSequence.length}/${CONFIG.N_FRAMES} frames`);
        }
        
        // Check if we have enough frames for prediction
        const now = Date.now();
        if (this.keypointSequence.length >= CONFIG.N_FRAMES && 
            (now - this.lastPredictionTime) > CONFIG.PREDICTION_THROTTLE_MS) {
            
            this.predictionsAttempted++;
            this.lastPredictionTime = now;
            
            // Create a copy of the sequence for prediction
            const sequenceToPredict = [...this.keypointSequence];
            
            // Keep some frames for continuity (sliding window)
            this.keypointSequence = this.keypointSequence.slice(-10);
            
            console.log(`🎯 Prediction attempt #${this.predictionsAttempted}:`, {
                sequenceLength: sequenceToPredict.length,
                timestamp: now
            });
            
            // Send for prediction
            if (this.onPredictionReady) {
                this.onPredictionReady(sequenceToPredict);
            } else {
                console.error('❌ No prediction callback set!');
            }
        }
    }

    drawResults(results) {
        // Clear canvas
        this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Flip horizontally for mirror effect
        this.canvasCtx.save();
        this.canvasCtx.scale(-1, 1);
        this.canvasCtx.translate(-this.canvas.width, 0);
        
        // Draw video frame
        this.canvasCtx.drawImage(results.image, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw landmarks
        this.drawLandmarks(results);
        
        this.canvasCtx.restore();
    }

    drawLandmarks(results) {
        // Draw pose connections (matching your OpenCV code)
        if (results.poseLandmarks) {
            // Draw shoulder-to-elbow and elbow-to-wrist connections
            const connections = [
                [11, 13], [13, 15], // Left arm
                [12, 14], [14, 16]  // Right arm
            ];
            
            connections.forEach(([start, end]) => {
                if (results.poseLandmarks[start] && results.poseLandmarks[end]) {
                    const startPoint = results.poseLandmarks[start];
                    const endPoint = results.poseLandmarks[end];
                    
                    this.canvasCtx.beginPath();
                    this.canvasCtx.moveTo(startPoint.x * this.canvas.width, startPoint.y * this.canvas.height);
                    this.canvasCtx.lineTo(endPoint.x * this.canvas.width, endPoint.y * this.canvas.height);
                    this.canvasCtx.strokeStyle = '#00FF00';
                    this.canvasCtx.lineWidth = 4;
                    this.canvasCtx.stroke();
                }
            });
            
            // Draw pose landmarks
            drawLandmarks(this.canvasCtx, results.poseLandmarks, { 
                color: '#FF0000', 
                lineWidth: 2,
                radius: 3 
            });
        }
        
        // Draw hand landmarks
        if (results.leftHandLandmarks) {
            drawConnectors(this.canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { 
                color: '#CC0000', 
                lineWidth: 5 
            });
            drawLandmarks(this.canvasCtx, results.leftHandLandmarks, { 
                color: '#FF0000', 
                lineWidth: 2,
                radius: 3 
            });
        }
        
        if (results.rightHandLandmarks) {
            drawConnectors(this.canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { 
                color: '#00CC00', 
                lineWidth: 5 
            });
            drawLandmarks(this.canvasCtx, results.rightHandLandmarks, { 
                color: '#FFFF00', 
                lineWidth: 2,
                radius: 3 
            });
        }
    }

    extractKeypoints(results) {
        // Reset detection counters
        this.landmarksDetected.leftHand = 0;
        this.landmarksDetected.rightHand = 0;
        this.landmarksDetected.pose = 0;
        
        const extractLandmarks = (landmarks, count, handName) => {
            if (!landmarks || landmarks.length === 0) {
                return new Array(count * 3).fill(0);
            }
            
            // Count detected landmarks
            this.landmarksDetected[handName] = landmarks.length;
            
            return landmarks.flatMap(landmark => [landmark.x, landmark.y, landmark.z]);
        };
        
        // Extract pose landmarks (6 keypoints: shoulders, elbows, wrists)
        const poseCoords = [];
        CONFIG.POSE_INDICES.forEach(index => {
            if (results.poseLandmarks && results.poseLandmarks[index]) {
                const landmark = results.poseLandmarks[index];
                poseCoords.push(landmark.x, landmark.y, landmark.z);
                this.landmarksDetected.pose++;
            } else {
                poseCoords.push(0, 0, 0);
            }
        });
        
        // Extract hand landmarks
        const leftHandCoords = extractLandmarks(results.leftHandLandmarks, 21, 'leftHand');
        const rightHandCoords = extractLandmarks(results.rightHandLandmarks, 21, 'rightHand');
        
        // Combine all keypoints
        const keypoints = [...leftHandCoords, ...rightHandCoords, ...poseCoords];
        
        // Debug logging (first frame and periodically)
        if (this.framesProcessed === 1 || this.framesProcessed % 60 === 0) {
            console.log('🔍 Keypoint Extraction Debug:', {
                totalLength: keypoints.length,
                leftHandDetected: this.landmarksDetected.leftHand,
                rightHandDetected: this.landmarksDetected.rightHand,
                poseDetected: this.landmarksDetected.pose,
                keypointRange: `[${Math.min(...keypoints).toFixed(3)} - ${Math.max(...keypoints).toFixed(3)}]`,
                hasNaN: keypoints.some(val => isNaN(val))
            });
        }
        
        return keypoints;
    }

    validateSequence(sequence) {
        if (!Array.isArray(sequence)) {
            console.error('❌ Sequence is not an array');
            return false;
        }
        
        if (sequence.length !== CONFIG.N_FRAMES) {
            console.error(`❌ Sequence length mismatch: ${sequence.length} vs ${CONFIG.N_FRAMES}`);
            return false;
        }
        
        for (let i = 0; i < sequence.length; i++) {
            const frame = sequence[i];
            if (!Array.isArray(frame) || frame.length !== 144) {
                console.error(`❌ Frame ${i} shape mismatch: ${frame.length} vs 144`);
                return false;
            }
            
            // Check for invalid values
            if (frame.some(val => isNaN(val) || !isFinite(val))) {
                console.error(`❌ Frame ${i} contains invalid values`);
                return false;
            }
        }
        
        console.log('✅ Sequence validation passed');
        return true;
    }

    handleError(error) {
        console.error('❌ MediaPipe Holistic Error:', error);
        
        const errorMap = {
            'MEDIAPIPE_INIT_ERROR': 'Failed to initialize MediaPipe',
            'CAMERA_ACCESS_ERROR': 'Cannot access camera. Please check permissions.',
            'VIDEO_PLAYBACK_ERROR': 'Error playing video stream'
        };
        
        const userMessage = errorMap[error] || `MediaPipe error: ${error}`;
        
        if (this.onError) {
            this.onError(userMessage);
        }
    }

    setPredictionCallback(callback) {
        this.onPredictionReady = callback;
        console.log('✅ Prediction callback set');
    }

    setErrorCallback(callback) {
        this.onError = callback;
        console.log('✅ Error callback set');
    }

    isCapturing() {
        return this.capturing;
    }

    getStats() {
        return {
            framesProcessed: this.framesProcessed,
            predictionsAttempted: this.predictionsAttempted,
            currentSequenceLength: this.keypointSequence.length,
            landmarksDetected: this.landmarksDetected,
            isCapturing: this.capturing
        };
    }

    cleanup() {
        console.log('🧹 Cleaning up MediaPipe manager...');
        this.stopCapture();
        
        if (this.holistic) {
            this.holistic.close();
            console.log('✅ MediaPipe Holistic closed');
        }
    }
}