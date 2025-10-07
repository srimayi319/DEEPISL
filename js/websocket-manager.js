class WebSocketManager {
    constructor() {
        this.socket = null;
        this.eventHandlers = new Map();
        this.isConnected = false;
    }

    initialize() {
        if (this.socket && this.isConnected) return;
        
        try {
            this.socket = io();
            this.setupEventListeners();
            console.log('WebSocket manager initialized');
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
        }
    }

    setupEventListeners() {
        this.socket.on('connect', () => this.handleConnect());
        this.socket.on('disconnect', () => this.handleDisconnect());
        this.socket.on('prediction_result', (data) => this.handlePredictionResult(data));
        this.socket.on('prediction_error', (data) => this.handlePredictionError(data));
        this.socket.on('animation_result', (data) => this.handleAnimationResult(data));
        this.socket.on('animation_error', (data) => this.handleAnimationError(data));
    }

    handleConnect() {
        console.log('Connected to server');
        this.isConnected = true;
        this.updateConnectionStatus('Connected', 'connected');
        this.emitEvent('statusUpdate', { status: 'READY' });
    }

    handleDisconnect() {
        console.log('Disconnected from server');
        this.isConnected = false;
        this.updateConnectionStatus('Disconnected', 'disconnected');
        this.emitEvent('statusUpdate', { status: 'DISCONNECTED' });
    }

    handlePredictionResult(data) {
        this.emitEvent('predictionResult', data);
    }

    handlePredictionError(data) {
        console.error('Prediction error:', data.error);
        this.emitEvent('predictionError', data);
        this.emitEvent('statusUpdate', { status: 'ERROR' });
    }

    handleAnimationResult(data) {
        this.emitEvent('animationResult', data);
    }

    handleAnimationError(data) {
        this.emitEvent('animationError', data);
    }

    updateConnectionStatus(text, className) {
        const element = document.getElementById(ELEMENTS.CONNECTION_STATUS);
        if (element) {
            element.textContent = text;
            element.className = `connection-status ${className}`;
        }
    }

    // Event handling system
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    emitEvent(event, data) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            handlers.forEach(handler => handler(data));
        }
    }

    // Socket emission methods
    predictSequence(sequence, history) {
        if (this.socket && this.isConnected) {
            this.socket.emit('predict_sequence', { sequence, history });
            return true;
        }
        console.warn('Cannot predict sequence: WebSocket not connected');
        return false;
    }

    generateAnimation(text) {
        if (this.socket && this.isConnected) {
            this.socket.emit('generate_animation', { text });
            return true;
        }
        console.warn('Cannot generate animation: WebSocket not connected');
        return false;
    }

    clearHistory() {
        if (this.socket && this.isConnected) {
            this.socket.emit('clear_history');
            return true;
        }
        return false;
    }

    getConnectionStatus() {
        return this.isConnected;
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.isConnected = false;
        }
    }
}