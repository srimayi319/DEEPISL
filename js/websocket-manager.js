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
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
        }
    }

    setupEventListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            this.updateConnectionStatus('Connected', 'connected');
            this.emitEvent('statusUpdate', { status: 'READY' });
        });
        this.socket.on('disconnect', () => {
            console.log('Disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('Disconnected', 'disconnected');
            this.emitEvent('statusUpdate', { status: 'DISCONNECTED' });
        });
        
        this.socket.on('prediction_result', (data) => this.emitEvent('predictionResult', data));
        this.socket.on('prediction_error', (data) => this.emitEvent('predictionError', data));
        this.socket.on('animation_result', (data) => this.emitEvent('animationResult', data));
        this.socket.on('animation_error', (data) => this.emitEvent('animationError', data));
    }

    on(event, handler) {
        if (!this.eventHandlers.has(event)) this.eventHandlers.set(event, []);
        this.eventHandlers.get(event).push(handler);
    }

    emitEvent(event, data) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) handlers.forEach(handler => handler(data));
    }

    predictSequence(sequence) {
        if (this.socket && this.isConnected) {
            this.socket.emit('predict_sequence', { sequence });
            return true;
        }
        return false;
    }

    generateAnimation(text) {
        if (this.socket && this.isConnected) {
            this.socket.emit('generate_animation', { text });
            return true;
        }
        return false;
    }

    clearHistory() {
        if (this.socket && this.isConnected) {
            this.socket.emit('clear_history');
        }
    }

    clearPredictionBuffer() {
        if (this.socket && this.isConnected) {
            this.socket.emit('clear_prediction_buffer');
        }
    }

    getConnectionStatus() {
        return this.isConnected;
    }

    updateConnectionStatus(text, className) {
        const el = document.getElementById(ELEMENTS.CONNECTION_STATUS);
        if (el) {
            el.textContent = text;
            el.className = `connection-status ${className}`;
        }
    }
}