class AnimationManager {
    constructor(webSocketManager, uiManager) {
        this.webSocketManager = webSocketManager;
        this.uiManager = uiManager;
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.webSocketManager.on('animationResult', (data) => this.handleAnimationResult(data));
        this.webSocketManager.on('animationError', (error) => this.handleAnimationError(error));
    }

    generateAnimation() {
        const text = this.uiManager.getTextInput();
        if (!text) {
            alert("Please enter some text to generate an animation.");
            return false;
        }

        this.uiManager.setAnimationLoading(true);

        if (this.webSocketManager.getConnectionStatus()) {
            const success = this.webSocketManager.generateAnimation(text);
            if (!success) {
                this.handleAnimationError('Failed to send animation request');
            }
            return success;
        } else {
            this.handleAnimationError('Not connected to the server.');
            return false;
        }
    }

    handleAnimationResult(data) {
        this.uiManager.setAnimationLoading(false);
        
        if (data.video_url) {
            this.uiManager.showAnimationVideo(data.video_url);
        } else {
            this.handleAnimationError('No video URL received from server');
        }
    }

    handleAnimationError(error) {
        console.error('Animation error:', error);
        this.uiManager.showAnimationError(error);
    }

    // Utility method to validate text input
    validateText(text) {
        if (!text || text.trim().length === 0) {
            return { valid: false, error: 'Text cannot be empty' };
        }
        
        if (text.length > 500) {
            return { valid: false, error: 'Text too long (max 500 characters)' };
        }
        
        // Basic sanitization
        const sanitizedText = text.replace(/[<>]/g, '');
        
        return { valid: true, text: sanitizedText };
    }
}