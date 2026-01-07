// CONFIGURATION
const CONFIG = {
    N_FRAMES: 30,
    MIN_CONFIDENCE: 0.65,
    
    // Motion Threshold: 
    // OpenCV used 0.01. 
    // Browsers are often noisier. We start with 0.02 to prevent false triggers.
    // If signs are not triggering, lower this slightly to 0.015.
    // If it triggers too often on nothing, raise it to 0.025.
    MOTION_THRESHOLD: 0.02,
    
    // Pose indices exactly matching Python OpenCV script
    POSE_INDICES: [11, 12, 13, 14, 15, 16]
};

// DOM ELEMENT IDs
const ELEMENTS = {
    START_BTN: 'start-btn',
    STOP_BTN: 'stop-btn',
    CLEAR_BTN: 'clear-btn',
    GENERATE_BTN: 'generate-btn',
    CONSTRUCT_SENTENCE_BTN: 'construct-sentence-btn',
    
    LIVE_CANVAS: 'live-canvas',
    DETECTED_TEXT: 'detected-text',
    TEXT_INPUT: 'text-input',
    
    STATUS_VALUE: 'status-value',
    CONFIDENCE_METER: 'confidence-meter',
    CONFIDENCE_BAR: 'confidence-bar',
    CONFIDENCE_TEXT: 'confidence-text',
    
    RECENT_SIGNS: 'detected-signs',
    WORD_COUNT: 'word-count',
    
    CONNECTION_STATUS: 'connection-status',
    
    ISL_TO_TEXT_BTN: 'isl-to-text-btn',
    TEXT_TO_ISL_BTN: 'text-to-isl-btn',
    ISL_TO_TEXT_SECTION: 'isl-to-text-section',
    TEXT_TO_ISL_SECTION: 'text-to-isl-section',
    
    ANIMATION_VIDEO: 'animation-video',
    ANIMATION_TEXT_PLACEHOLDER: 'animation-text-placeholder',
    VIDEO_PLACEHOLDER: 'video-placeholder'
};