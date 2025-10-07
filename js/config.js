// Application Configuration
const CONFIG = {
    // MediaPipe Settings
    N_FRAMES: 30,
    FRAME_SKIP: 4,
    MIN_CONFIDENCE: 0.65,
    PREDICTION_THROTTLE_MS: 400,
    
    // MediaPipe Holistic Options
    HOLISTIC_OPTIONS: {
        modelComplexity: 0,
        smoothLandmarks: false,
        enableSegmentation: false,
        smoothSegmentationMask: false,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    },
    
    // Pose landmarks to extract (reduced for efficiency)
    POSE_INDICES: [11, 12, 13, 14, 15, 16]
};

// DOM Element IDs
const ELEMENTS = {
    // Mode switching
    ISL_TO_TEXT_BTN: 'isl-to-text-btn',
    TEXT_TO_ISL_BTN: 'text-to-isl-btn',
    ISL_TO_TEXT_SECTION: 'isl-to-text-section',
    TEXT_TO_ISL_SECTION: 'text-to-isl-section',
    
    // Video & Controls
    START_BTN: 'start-btn',
    STOP_BTN: 'stop-btn',
    CLEAR_BTN: 'clear-btn',
    LIVE_CANVAS: 'live-canvas',
    VIDEO_PLACEHOLDER: 'video-placeholder',
    
    // Status & Output
    STATUS_VALUE: 'status-value',
    DETECTED_TEXT: 'detected-text',
    CONFIDENCE_METER: 'confidence-meter',
    CONFIDENCE_BAR: 'confidence-bar',
    CONFIDENCE_TEXT: 'confidence-text',
    RECENT_SIGNS: 'recent-signs',
    WORD_COUNT: 'word-count',
    
    // Animation
    TEXT_INPUT: 'text-input',
    GENERATE_BTN: 'generate-btn',
    ANIMATION_VIDEO: 'animation-video',
    ANIMATION_TEXT_PLACEHOLDER: 'animation-text-placeholder',
    
    // Connection
    CONNECTION_STATUS: 'connection-status'
};