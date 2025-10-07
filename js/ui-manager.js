class UIManager {
    constructor() {
        this.elements = {};
        this.currentMode = 'isl-to-text';
        this.initializeElements();
    }

    initializeElements() {
        // Cache DOM elements for better performance
        Object.keys(ELEMENTS).forEach(key => {
            const elementId = ELEMENTS[key];
            this.elements[elementId] = document.getElementById(elementId);
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        if (mode === 'isl-to-text') {
            this.showISLToTextMode();
        } else {
            this.showTextToISLMode();
        }
    }

    showISLToTextMode() {
        this.elements[ELEMENTS.ISL_TO_TEXT_SECTION].classList.remove('hidden');
        this.elements[ELEMENTS.TEXT_TO_ISL_SECTION].classList.add('hidden');
        
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.add('bg-white', 'text-gray-800', 'shadow-md');
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.remove('text-gray-600');
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.remove('bg-white', 'text-gray-800', 'shadow-md');
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.add('text-gray-600');
    }

    showTextToISLMode() {
        this.elements[ELEMENTS.ISL_TO_TEXT_SECTION].classList.add('hidden');
        this.elements[ELEMENTS.TEXT_TO_ISL_SECTION].classList.remove('hidden');
        
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.add('bg-white', 'text-gray-800', 'shadow-md');
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.remove('text-gray-600');
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.remove('bg-white', 'text-gray-800', 'shadow-md');
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.add('text-gray-600');
    }

    updateStatus(status, type = '') {
        const statusValueElement = this.elements[ELEMENTS.STATUS_VALUE];
        const statusBox = statusValueElement.parentElement;
        
        statusValueElement.textContent = status;
        
        // Reset classes
        statusBox.className = 'status-box';
        if (type) {
            statusBox.classList.add(type);
        }
    }

    updateConfidence(confidence) {
        const confidencePercent = Math.round(confidence * 100);
        
        this.elements[ELEMENTS.CONFIDENCE_METER].classList.remove('hidden');
        this.elements[ELEMENTS.CONFIDENCE_BAR].style.width = `${confidencePercent}%`;
        this.elements[ELEMENTS.CONFIDENCE_TEXT].textContent = `Confidence: ${confidencePercent}%`;
        
        return confidencePercent;
    }

    updateDetectedText(sentence) {
        this.elements[ELEMENTS.DETECTED_TEXT].value = sentence;
        this.updateWordCount();
    }

    updateWordCount() {
        const text = this.elements[ELEMENTS.DETECTED_TEXT].value;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        this.elements[ELEMENTS.WORD_COUNT].textContent = `${words} word${words !== 1 ? 's' : ''}`;
    }

    updateRecentSigns(signHistory) {
        if (!signHistory || signHistory.length === 0) {
            this.elements[ELEMENTS.RECENT_SIGNS].innerHTML = 'No signs detected yet...';
            return;
        }
        
        const recent = signHistory.slice(-5);
        this.elements[ELEMENTS.RECENT_SIGNS].innerHTML = recent.map(sign => 
            `<span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-full mr-2">${sign}</span>`
        ).join('');
    }

    showVideoPlaceholder(show) {
        if (show) {
            this.elements[ELEMENTS.VIDEO_PLACEHOLDER].classList.remove('hidden');
        } else {
            this.elements[ELEMENTS.VIDEO_PLACEHOLDER].classList.add('hidden');
        }
    }

    // Animation section methods
    setAnimationLoading(loading) {
        if (loading) {
            this.elements[ELEMENTS.ANIMATION_VIDEO].classList.add('hidden');
            this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].textContent = 'Generating animation...';
            this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].classList.remove('hidden');
            this.elements[ELEMENTS.GENERATE_BTN].disabled = true;
            this.elements[ELEMENTS.GENERATE_BTN].textContent = 'Generating...';
        } else {
            this.elements[ELEMENTS.GENERATE_BTN].disabled = false;
            this.elements[ELEMENTS.GENERATE_BTN].textContent = '✨ Generate Animation';
        }
    }

    showAnimationVideo(videoUrl) {
        this.elements[ELEMENTS.ANIMATION_VIDEO].src = videoUrl;
        this.elements[ELEMENTS.ANIMATION_VIDEO].onloadeddata = () => {
            this.elements[ELEMENTS.ANIMATION_VIDEO].classList.remove('hidden');
            this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].classList.add('hidden');
            this.elements[ELEMENTS.ANIMATION_VIDEO].play();
        };
        
        this.elements[ELEMENTS.ANIMATION_VIDEO].onerror = () => {
            this.showAnimationError('Failed to load animation video');
        };
    }

    showAnimationError(error) {
        this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].textContent = `Error: ${error}`;
        this.elements[ELEMENTS.ANIMATION_VIDEO].classList.add('hidden');
        this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].classList.remove('hidden');
        this.setAnimationLoading(false);
    }

    getTextInput() {
        return this.elements[ELEMENTS.TEXT_INPUT].value.trim();
    }

    clearTextInput() {
        this.elements[ELEMENTS.TEXT_INPUT].value = '';
    }

    clearDetectedText() {
        this.elements[ELEMENTS.DETECTED_TEXT].value = '';
        this.updateWordCount();
    }

    // Getter methods for elements (for event listeners)
    getElement(id) {
        return this.elements[id];
    }

    getCurrentMode() {
        return this.currentMode;
    }
}