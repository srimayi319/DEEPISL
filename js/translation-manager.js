class TranslationManager {
    constructor(webSocketManager, uiManager) {
        this.webSocketManager = webSocketManager;
        this.uiManager = uiManager;
        this.signHistory = [];
        this.setupEventListeners();
    }

    setupEventListeners() {
        console.log("TranslationManager: Setting up listeners.");
        this.webSocketManager.on('predictionResult', (data) => {
            console.log("TranslationManager: Received PREDICTION RESULT:", data);
            this.handlePredictionResult(data)
        });
        this.webSocketManager.on('predictionError', (data) => {
            console.error("TranslationManager: Received PREDICTION ERROR:", data);
            this.handlePredictionError(data)
        });
        this.webSocketManager.on('statusUpdate', (data) => {
            console.log("TranslationManager: Received STATUS UPDATE:", data);
            this.handleStatusUpdate(data)
        });
    }

    handlePredictionResult(data) {
        const { label, confidence, history } = data;
        console.log(`TranslationManager: Label: ${label}, Confidence: ${confidence}`);
        
        this.uiManager.updateConfidence(confidence);

        if (label) {
            this.uiManager.updateStatus(`${label.toUpperCase()}`, 'confident');
        } else {
            this.uiManager.updateStatus('DETECTING...', 'uncertain');
        }

        this.signHistory = history || [];

        console.log(`TranslationManager: Updating Detected Signs to:`, history);
        this.uiManager.updateDetectedSigns(this.signHistory);
    }

    handlePredictionError(data) {
        console.error('TranslationManager: Prediction error:', data?.error);
        this.uiManager.updateStatus('PREDICTION_ERROR', 'error');
    }

    handleStatusUpdate(data) {
        if (data?.status) {
            this.uiManager.updateStatus(data.status);
        }
    }

    clearHistory() {
        console.log("TranslationManager: Clearing History");
        this.signHistory = [];
        this.uiManager.updateDetectedSigns([]);
        this.uiManager.clearDetectedText();
        this.webSocketManager.clearHistory();
    }

    getSignHistory() {
        return this.signHistory;
    }
}