class MediaPipeManager {
    constructor() {
        this.hands = null;
        this.pose = null;
        this.capturing = false;

        this.video = null;
        this.canvas = null;
        this.canvasCtx = null;

        this.keypointSequence = [];
        this.prev_keypoints = null;

        this.motionStarted = false;
        this.isCooldown = false;
        this.framesCollected = 0;

        this.onPredictionReady = null;
        this.onStateChange = null;
        this.onClearBuffer = null;

        // --- FIX: Declare these here so processFrame() can see them ---
        this.lastHands = null;
        this.lastPose = null;
    }

    async initialize() {
        try {
            this.video = document.createElement('video');
            this.canvas = document.getElementById(ELEMENTS.LIVE_CANVAS);
            this.canvasCtx = this.canvas.getContext('2d');

            // ===== Hands =====
            this.hands = new Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
            });

            this.hands.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            // ===== Pose =====
            this.pose = new Pose({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
            });

            this.pose.setOptions({
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            // Store results in these variables
            this.hands.onResults((res) => this.lastHands = res);
            this.pose.onResults((res) => this.lastPose = res);

            console.log("✅ MediaPipe Hands + Pose initialized");
            return true;

        } catch (err) {
            console.error("❌ MediaPipe init error:", err);
            return false;
        }
    }

    async startCapture() {
        if (this.capturing) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, frameRate: 30 }
            });

            this.video.srcObject = stream;
            this.video.play();

            this.canvas.width = 640;
            this.canvas.height = 480;

            this.capturing = true;
            this.keypointSequence = [];
            this.prev_keypoints = null;
            this.motionStarted = false;
            this.isCooldown = false;

            this.processFrame();
        } catch (err) {
            console.error("Camera Error:", err);
        }
    }

    stopCapture() {
        this.capturing = false;
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
        }
    }

    async processFrame() {
        if (!this.capturing) return;

        await this.hands.send({ image: this.video });
        await this.pose.send({ image: this.video });

        if (this.lastHands && this.lastPose) {
            this.onResults(this.lastHands, this.lastPose);
        }

        requestAnimationFrame(() => this.processFrame());
    }

    onResults(handResults, poseResults) {
        this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.canvasCtx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        if (typeof drawConnectors !== 'undefined') {
            if (handResults.multiHandLandmarks) {
                for (const lm of handResults.multiHandLandmarks) {
                    drawConnectors(this.canvasCtx, lm, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
                }
            }

            if (poseResults.poseLandmarks) {
                drawConnectors(this.canvasCtx, poseResults.poseLandmarks, POSE_CONNECTIONS, { color: '#FF0000', lineWidth: 2 });
            }
        }

        const keypoints = this.extractKeypoints(handResults, poseResults);
        const motion = this.calculateMotion(keypoints, this.prev_keypoints);
        this.prev_keypoints = keypoints;

        this.canvasCtx.fillStyle = "white";
        this.canvasCtx.font = "16px Arial";
        this.canvasCtx.fillText(`Motion: ${motion.toFixed(4)}`, 20, 30);

        this.runStateMachine(keypoints, motion);
    }

    extractKeypoints(handResults, poseResults) {
        let left = new Array(63).fill(0);
        let right = new Array(63).fill(0);

        if (handResults.multiHandLandmarks && handResults.multiHandedness) {
            for (let i = 0; i < handResults.multiHandLandmarks.length; i++) {
                const lm = handResults.multiHandLandmarks[i];
                const label = handResults.multiHandedness[i].label;
                const flat = lm.flatMap(p => [p.x, p.y, p.z]);
                if (label === "Left") left = flat;
                else right = flat;
            }
        }

        let pose = new Array(18).fill(0);
        if (poseResults.poseLandmarks) {
            pose = CONFIG.POSE_INDICES.flatMap(i => {
                const p = poseResults.poseLandmarks[i];
                return [p.x, p.y, p.z];
            });
        }

        return [...left, ...right, ...pose];
    }

    calculateMotion(curr, prev) {
        if (!prev) return 0;
        let sum = 0;
        for (let i = 0; i < curr.length; i++) {
            sum += (curr[i] - prev[i]) ** 2;
        }
        return Math.sqrt(sum);
    }

    runStateMachine(keypoints, motion) {
        if (!this.motionStarted) {
            if (this.isCooldown) return;

            if (motion > CONFIG.MOTION_THRESHOLD) {
                if (this.onClearBuffer) this.onClearBuffer();
                this.motionStarted = true;
                this.keypointSequence = [];
                this.framesCollected = 0;
                this.notifyState("RECORDING");
            } else {
                this.notifyState("WAITING");
            }

        } else {
            // --- MOTION GATE ---
            // Ensure motion doesn't drop too low (prevents static frames)
            if (motion < (CONFIG.MOTION_THRESHOLD / 2)) {
                 this.motionStarted = false;
                 this.keypointSequence = [];
                 this.framesCollected = 0;
                 console.log("🔴 Motion lost during capture");
                 return;
            }

            this.keypointSequence.push(keypoints);
            this.framesCollected++;

            if (this.framesCollected >= CONFIG.N_FRAMES) {
                // --- FIX: Sanity check before sending ---
                if (this.keypointSequence.length > 0 && this.keypointSequence[0].length === 144) {
                    const seq = [...this.keypointSequence];
                    this.onPredictionReady(seq);
                } else {
                    console.error("❌ Invalid sequence shape sent to predictor");
                }

                this.motionStarted = false;
                this.keypointSequence = [];
                this.framesCollected = 0;
                this.isCooldown = true;
                this.notifyState("COOLDOWN");

                setTimeout(() => this.isCooldown = false, 2000);
            }
        }
    }

    notifyState(state) {
        if (this.onStateChange) this.onStateChange(state);
    }

    setPredictionCallback(cb) { this.onPredictionReady = cb; }
    setStateChangeCallback(cb) { this.onStateChange = cb; }
    setClearBufferCallback(cb) { this.onClearBuffer = cb; }
}