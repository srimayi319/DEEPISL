class TranslationManager {
    constructor(webSocketManager, uiManager) {
        this.webSocketManager = webSocketManager;
        this.uiManager = uiManager;
        this.signHistory = [];
        this.setupEventListeners();
    }

    setupEventListeners() {
        // WebSocket events
        this.webSocketManager.on('predictionResult', (data) => this.handlePredictionResult(data));
        this.webSocketManager.on('predictionError', (data) => this.handlePredictionError(data));
        this.webSocketManager.on('statusUpdate', (data) => this.handleStatusUpdate(data));
    }

    handlePredictionResult(data) {
        const { label, confidence, sentence, history } = data;
        
        // Update confidence display
        const confidencePercent = this.uiManager.updateConfidence(confidence);
        
        // Update based on confidence level
        if (confidence > CONFIG.MIN_CONFIDENCE) {
            this.uiManager.updateStatus(`${label.toUpperCase()} (${confidencePercent}%)`, 'confident');
            
            // Update history and display
            this.signHistory = history || [];
            
            if (sentence) {
                this.uiManager.updateDetectedText(sentence);
            }
            
            this.uiManager.updateRecentSigns(this.signHistory);
        } else {
            this.uiManager.updateStatus('DETECTING...', 'uncertain');
        }
    }

    handlePredictionError(data) {
        console.error('Prediction error:', data.error);
        this.uiManager.updateStatus('PREDICTION_ERROR', 'error');
    }

    handleStatusUpdate(data) {
        this.uiManager.updateStatus(data.status);
    }

    // Method called by MediaPipe when sequence is ready
    handleSequenceReady(sequence) {
        if (this.webSocketManager.getConnectionStatus()) {
            const success = this.webSocketManager.predictSequence(sequence, this.signHistory);
            if (!success) {
                this.uiManager.updateStatus('WEBSOCKET_ERROR', 'error');
            }
        } else {
            this.uiManager.updateStatus('DISCONNECTED', 'error');
        }
    }

    clearHistory() {
        this.signHistory = [];
        this.uiManager.updateRecentSigns(this.signHistory);
        this.uiManager.clearDetectedText();
        this.webSocketManager.clearHistory();
    }

    getSignHistory() {
        return this.signHistory;
    }

    setSignHistory(history) {
        this.signHistory = history || [];
    }
}