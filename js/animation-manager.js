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
        // FIX: Handle both string errors and object errors
        let errorMessage = "Unknown error";
        
        if (typeof error === 'string') {
            errorMessage = error;
        } else if (error.error) {
            errorMessage = error.error;
        } else if (error.message) {
            errorMessage = error.message;
        } else {
            errorMessage = JSON.stringify(error);
        }

        console.error('Animation error:', errorMessage);
        this.uiManager.showAnimationError(errorMessage);
    }
}