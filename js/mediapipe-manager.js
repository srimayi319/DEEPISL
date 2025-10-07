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
    }

    async initialize() {
        try {
            this.video = document.createElement('video');
            this.canvas = document.getElementById(ELEMENTS.LIVE_CANVAS);
            this.canvasCtx = this.canvas.getContext('2d');

            this.holistic = new Holistic({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
            });

            this.holistic.setOptions(CONFIG.HOLISTIC_OPTIONS);
            this.holistic.onResults((results) => this.onResults(results));
            this.holistic.onError((error) => this.handleError(error));
            
            console.log('MediaPipe manager initialized');
            return true;
        } catch (error) {
            console.error('Failed to initialize MediaPipe:', error);
            this.handleError('MEDIAPIPE_INIT_ERROR');
            return false;
        }
    }

    async startCapture() {
        if (this.capturing) {
            console.log('Already capturing');
            return false;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            this.video.srcObject = stream;
            this.capturing = true;
            this.keypointSequence = [];
            this.frameCounter = 0;
            
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    this.processVideoFrame();
                    resolve(true);
                };
                
                this.video.onerror = () => {
                    this.handleError('VIDEO_PLAYBACK_ERROR');
                    resolve(false);
                };
            });

        } catch (err) {
            console.error("Error accessing webcam: ", err);
            this.handleError('CAMERA_ACCESS_ERROR');
            return false;
        }
    }

    stopCapture() {
        this.capturing = false;
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    async processVideoFrame() {
        if (!this.capturing) return;
        
        this.frameCounter++;
        if (this.frameCounter % CONFIG.FRAME_SKIP === 0) {
            try {
                await this.holistic.send({ image: this.video });
            } catch (error) {
                console.error('Error processing frame:', error);
            }
        }
        
        requestAnimationFrame(() => this.processVideoFrame());
    }

    onResults(results) {
        // Draw landmarks
        this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.canvasCtx.save();
        this.canvasCtx.scale(-1, 1);
        this.canvasCtx.translate(-this.canvas.width, 0);
        this.canvasCtx.drawImage(results.image, 0, 0, this.canvas.width, this.canvas.height);
        
        this.drawLandmarks(results);
        this.canvasCtx.restore();

        // Extract and store keypoints
        const keypoints = this.extractKeypoints(results);
        this.keypointSequence.push(keypoints);
        
        // Throttle predictions
        const now = Date.now();
        if (this.keypointSequence.length >= CONFIG.N_FRAMES && 
            (now - this.lastPredictionTime) > CONFIG.PREDICTION_THROTTLE_MS) {
            
            this.lastPredictionTime = now;
            const sequenceToPredict = [...this.keypointSequence];
            this.keypointSequence.length = 0; // Clear the array
            
            if (this.onPredictionReady) {
                this.onPredictionReady(sequenceToPredict);
            }
        }
    }

    drawLandmarks(results) {
        if (results.poseLandmarks) {
            drawConnectors(this.canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
            drawLandmarks(this.canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
        }
        if (results.leftHandLandmarks) {
            drawConnectors(this.canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 5 });
            drawLandmarks(this.canvasCtx, results.leftHandLandmarks, { color: '#FF0000', lineWidth: 2 });
        }
        if (results.rightHandLandmarks) {
            drawConnectors(this.canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 5 });
            drawLandmarks(this.canvasCtx, results.rightHandLandmarks, { color: '#FFFF00', lineWidth: 2 });
        }
    }

    extractKeypoints(results) {
        const extract = (landmarks, num) => {
            if (!landmarks) return new Array(num * 3).fill(0);
            return landmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
        };
        
        const poseCoords = [];
        
        CONFIG.POSE_INDICES.forEach(idx => {
            if (results.poseLandmarks && results.poseLandmarks[idx]) {
                const lm = results.poseLandmarks[idx];
                poseCoords.push(lm.x, lm.y, lm.z);
            } else {
                poseCoords.push(0, 0, 0);
            }
        });
        
        const leftHandCoords = extract(results.leftHandLandmarks, 21);
        const rightHandCoords = extract(results.rightHandLandmarks, 21);
        
        return [...leftHandCoords, ...rightHandCoords, ...poseCoords];
    }

    handleError(error) {
        console.error('MediaPipe Holistic Error:', error);
        if (this.onError) {
            this.onError(error);
        }
    }

    setPredictionCallback(callback) {
        this.onPredictionReady = callback;
    }

    setErrorCallback(callback) {
        this.onError = callback;
    }

    isCapturing() {
        return this.capturing;
    }

    cleanup() {
        this.stopCapture();
        if (this.holistic) {
            this.holistic.close();
        }
    }
}