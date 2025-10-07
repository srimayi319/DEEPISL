class DeepISLApp {
    constructor() {
        this.webSocketManager = new WebSocketManager();
        this.uiManager = new UIManager();
        this.mediaPipeManager = new MediaPipeManager();
        this.translationManager = null;
        this.animationManager = null;
        
        this.initialize();
    }

    async initialize() {
        try {
            console.log('Initializing DeepISL App...');
            
            // Initialize managers in sequence
            await this.mediaPipeManager.initialize();
            this.webSocketManager.initialize();
            
            // Create dependent managers
            this.translationManager = new TranslationManager(this.webSocketManager, this.uiManager);
            this.animationManager = new AnimationManager(this.webSocketManager, this.uiManager);
            
            // Set up MediaPipe callbacks
            this.mediaPipeManager.setPredictionCallback(
                (sequence) => this.translationManager.handleSequenceReady(sequence)
            );
            
            this.mediaPipeManager.setErrorCallback(
                (error) => this.handleMediaPipeError(error)
            );
            
            // Set up event listeners
            this.setupEventListeners();
            
            console.log('DeepISL App initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.uiManager.updateStatus('INIT_ERROR', 'error');
        }
    }

    setupEventListeners() {
        // Mode switching
        this.uiManager.getElement(ELEMENTS.ISL_TO_TEXT_BTN).addEventListener('click', 
            () => this.uiManager.switchMode('isl-to-text'));
        this.uiManager.getElement(ELEMENTS.TEXT_TO_ISL_BTN).addEventListener('click', 
            () => this.uiManager.switchMode('text-to-isl'));

        // ISL to Text controls
        this.uiManager.getElement(ELEMENTS.START_BTN).addEventListener('click', 
            () => this.startSignDetection());
        this.uiManager.getElement(ELEMENTS.STOP_BTN).addEventListener('click', 
            () => this.stopSignDetection());
        this.uiManager.getElement(ELEMENTS.CLEAR_BTN).addEventListener('click', 
            () => this.translationManager.clearHistory());

        // Text to ISL controls
        this.uiManager.getElement(ELEMENTS.GENERATE_BTN).addEventListener('click', 
            () => this.animationManager.generateAnimation());

        // Window events
        window.addEventListener('resize', this.debounce(() => this.handleResize(), 250));
        document.addEventListener('visibilitychange', () => this.handleVisibilityChange());
        
        // Handle page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    async startSignDetection() {
        if (this.mediaPipeManager.isCapturing()) {
            console.log('Sign detection already active');
            return;
        }
        
        if (!this.webSocketManager.getConnectionStatus()) {
            alert('Not connected to server. Please wait for connection or refresh the page.');
            return;
        }
        
        this.uiManager.showVideoPlaceholder(false);
        this.uiManager.updateStatus('STARTING...');
        
        const success = await this.mediaPipeManager.startCapture();
        if (success) {
            this.uiManager.updateStatus('DETECTING...');
        } else {
            this.uiManager.updateStatus('CAMERA_ERROR', 'error');
            this.uiManager.showVideoPlaceholder(true);
        }
    }

    stopSignDetection() {
        this.mediaPipeManager.stopCapture();
        this.uiManager.updateStatus('IDLE');
        this.uiManager.showVideoPlaceholder(true);
    }

    handleResize() {
        if (this.mediaPipeManager.isCapturing() && this.mediaPipeManager.video.videoWidth) {
            this.mediaPipeManager.canvas.width = this.mediaPipeManager.video.videoWidth;
            this.mediaPipeManager.canvas.height = this.mediaPipeManager.video.videoHeight;
        }
    }

    handleVisibilityChange() {
        if (document.hidden && this.mediaPipeManager.isCapturing()) {
            this.stopSignDetection();
        }
    }

    handleMediaPipeError(error) {
        console.error('MediaPipe error:', error);
        this.uiManager.updateStatus('MEDIAPIPE_ERROR', 'error');
        this.stopSignDetection();
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    cleanup() {
        this.mediaPipeManager.cleanup();
        this.webSocketManager.disconnect();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.deepISLApp = new DeepISLApp();
});