class DeepISLApp {
    constructor() {
        this.wsManager = new WebSocketManager();
        this.mpManager = new MediaPipeManager();
        this.uiManager = new UIManager();
        this.translationManager = new TranslationManager(this.wsManager, this.uiManager);
        this.animationManager = new AnimationManager(this.wsManager, this.uiManager);
        this.signHistory = [];
        this.initialize();
    }

    async initialize() {
        this.uiManager.initializeElements();
        await this.mpManager.initialize();
        this.wsManager.initialize();

        this.mpManager.setPredictionCallback((seq) => this.handlePrediction(seq));
        this.mpManager.setStateChangeCallback((state) => this.handleMediaPipeState(state));
        this.mpManager.setClearBufferCallback(() => {
            this.wsManager.clearPredictionBuffer();
        });

        this.setupTabs();
        this.setupAnimationButton();
        this.setupMediaPipeControls();
        this.setupConstructSentenceButton();

        this.wsManager.on('predictionResult', (data) => this.handleResult(data));
    }

    setupTabs() {
        const islBtn = document.getElementById('isl-to-text-btn');
        const textBtn = document.getElementById('text-to-isl-btn');

        if (!islBtn || !textBtn) return;

        islBtn.addEventListener('click', () => {
            this.uiManager.switchMode("isl-to-text");
        });

        textBtn.addEventListener('click', () => {
            this.uiManager.switchMode("text-to-isl");
        });
    }

    setupAnimationButton() {
        const generateBtn = document.getElementById('generate-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => {
                this.animationManager.generateAnimation();
            });
        }
    }

    setupMediaPipeControls() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const clearBtn = document.getElementById('clear-btn');

        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.mpManager.startCapture();
                this.uiManager.updateStatus('INITIALIZING CAMERA...');
                this.signHistory = [];
                this.uiManager.updateDetectedSigns([]);
                this.uiManager.clearDetectedText();
            });
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                this.mpManager.stopCapture();
                this.uiManager.updateStatus('IDLE');
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.signHistory = [];
                this.translationManager.clearHistory();
                this.uiManager.clearDetectedText();
                this.uiManager.updateDetectedSigns([]);
                this.uiManager.updateStatus('CLEARED', 'uncertain');
            });
        }
    }

    setupConstructSentenceButton() {
        const btn = document.getElementById('construct-sentence-btn');
        if (btn) {
            btn.addEventListener('click', () => {
                this.uiManager.constructSentence();
            });
        }
    }

    handleMediaPipeState(state) {
        if (this.uiManager.getCurrentMode() !== 'isl-to-text') return;

        if (state === 'WAITING') {
            this.uiManager.updateStatus('WAITING FOR SIGN', 'uncertain');
        } else if (state === 'RECORDING') {
            this.uiManager.updateStatus('🔴 RECORDING...', 'recording');
        } else if (state === 'PROCESSING...') {
            this.uiManager.updateStatus('PREDICTING...', 'confident');
        } else if (state === 'COOLDOWN...') {
            this.uiManager.updateStatus('COOLDOWN...', 'uncertain');
        }
    }

    handlePrediction(sequence) {
        if (this.uiManager.getCurrentMode() !== 'isl-to-text') return;
        this.wsManager.predictSequence(sequence);
    }

    handleResult(data) {
        if (this.uiManager.getCurrentMode() !== 'isl-to-text') return;

        const { label, confidence, history } = data;
        
        this.uiManager.updateConfidence(confidence);
        
        if (confidence > CONFIG.MIN_CONFIDENCE) {
            this.signHistory = history; 
            this.uiManager.updateDetectedSigns(history);
            this.uiManager.updateStatus(`${label.toUpperCase()}`, 'confident');
        } else {
            this.uiManager.updateStatus('WAITING FOR SIGN', 'uncertain');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeepISLApp();
});