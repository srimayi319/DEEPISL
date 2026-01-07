class UIManager {
    constructor() {
        this.elements = {};
        this.currentMode = "isl-to-text";
        this.signs = [];
        this.initializeElements();
    }

    initializeElements() {
        Object.values(ELEMENTS).forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
        // Grab layout columns
        this.elements['input-column'] = document.getElementById('input-column');
        this.elements['output-column'] = document.getElementById('output-column');
    }

    switchMode(mode) {
        this.currentMode = mode;
        mode === "isl-to-text" ? this.showISLToTextMode() : this.showTextToISLMode();
    }

    showISLToTextMode() {
        this.elements[ELEMENTS.ISL_TO_TEXT_SECTION].classList.remove("hidden");
        this.elements[ELEMENTS.TEXT_TO_ISL_SECTION].classList.add("hidden");

        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.add("bg-white", "text-gray-800", "shadow-md");
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.remove("text-gray-600");
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.remove("bg-white", "text-gray-800", "shadow-md");
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.add("text-gray-600");

        if(this.elements['output-column']) this.elements['output-column'].classList.remove('hidden');
        if(this.elements['input-column']) {
            this.elements['input-column'].classList.remove('lg:col-span-2');
            this.elements['input-column'].classList.add('lg:col-span-1');
        }
    }

    showTextToISLMode() {
        this.elements[ELEMENTS.ISL_TO_TEXT_SECTION].classList.add("hidden");
        this.elements[ELEMENTS.TEXT_TO_ISL_SECTION].classList.remove("hidden");

        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.add("bg-white", "text-gray-800", "shadow-md");
        this.elements[ELEMENTS.TEXT_TO_ISL_BTN].classList.remove("text-gray-600");
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.remove("bg-white", "text-gray-800", "shadow-md");
        this.elements[ELEMENTS.ISL_TO_TEXT_BTN].classList.add("text-gray-600");

        if(this.elements['output-column']) this.elements['output-column'].classList.add('hidden');
        if(this.elements['input-column']) {
            this.elements['input-column'].classList.remove('lg:col-span-1');
            this.elements['input-column'].classList.add('lg:col-span-2');
        }
    }

    updateStatus(text, type = "") {
        const statusValue = this.elements[ELEMENTS.STATUS_VALUE];
        if (!statusValue) return;
        const statusBox = statusValue.parentElement;
        statusValue.textContent = text;
        statusBox.className = "status-box";
        if (type) statusBox.classList.add(type);
    }

    updateConfidence(confidence) {
        const percent = Math.round(confidence * 100);
        this.elements[ELEMENTS.CONFIDENCE_METER].classList.remove("hidden");
        this.elements[ELEMENTS.CONFIDENCE_BAR].style.width = `${percent}%`;
        this.elements[ELEMENTS.CONFIDENCE_TEXT].textContent = `Confidence: ${percent}%`;
        return percent;
    }

    updateDetectedText(text) {
        this.elements[ELEMENTS.DETECTED_TEXT].value = text;
    }

    clearDetectedText() {
        this.elements[ELEMENTS.DETECTED_TEXT].value = "";
    }

    updateDetectedSigns(history) {
        if (!history || history.length === 0) {
            this.elements[ELEMENTS.RECENT_SIGNS].innerHTML = "No signs detected yet...";
            this.disableConstructButton(true);
            this.signs = [];
            return;
        }

        this.signs = history; 
        this.elements[ELEMENTS.RECENT_SIGNS].innerHTML = history
            .map(sign =>
                `<span class="bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full font-medium mr-2 mb-2">
                    ${sign}
                </span>`
            )
            .join("");
        
        this.disableConstructButton(false);
    }

    constructSentence() {
        if (this.signs.length === 0) return;
        const sentence = this.signs.join(" ");
        this.updateDetectedText(sentence);
    }

    disableConstructButton(disabled) {
        const btn = this.elements[ELEMENTS.CONSTRUCT_SENTENCE_BTN];
        if (btn) {
            if (disabled) {
                btn.disabled = true;
                btn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                btn.disabled = false;
                btn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
    }

    showVideoPlaceholder(show) {
        if(this.elements[ELEMENTS.VIDEO_PLACEHOLDER]) {
            this.elements[ELEMENTS.VIDEO_PLACEHOLDER].classList.toggle("hidden", !show);
        }
    }

    setAnimationLoading(loading) {
        const btn = this.elements[ELEMENTS.GENERATE_BTN];
        if (loading) {
            btn.disabled = true;
            btn.textContent = "Generating...";
            this.elements[ELEMENTS.ANIMATION_VIDEO].classList.add("hidden");
            this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].classList.remove("hidden");
            this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].textContent = "Generating animation...";
        } else {
            btn.disabled = false;
            btn.textContent = "✨ Generate Animation";
        }
    }

    showAnimationVideo(url) {
    const video = this.elements[ELEMENTS.ANIMATION_VIDEO];
    const placeholder = this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER];

    console.log("🎬 Loading animation:", url);

    // Force reload even if same filename is generated twice
    video.pause();
    video.removeAttribute("src");
    video.load();

    // Cache-buster (important on cloud)
    const bustUrl = url + "?t=" + new Date().getTime();
    video.src = bustUrl;

    // Show video as soon as metadata is ready
    video.onloadedmetadata = () => {
        console.log("✅ Video metadata loaded");
        video.classList.remove("hidden");
        placeholder.classList.add("hidden");
        video.play().catch(() => {});
    };

    // Fallback if metadata is slow
    video.oncanplay = () => {
        console.log("▶️ Video can play");
        video.classList.remove("hidden");
        placeholder.classList.add("hidden");
    };

    video.onerror = () => {
        console.error("❌ Video failed to load:", video.src);
        this.showAnimationError("Failed to load animation file.");
    };
}


    showAnimationError(message) {
        this.elements[ELEMENTS.ANIMATION_VIDEO].classList.add("hidden");
        this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].classList.remove("hidden");
        this.elements[ELEMENTS.ANIMATION_TEXT_PLACEHOLDER].textContent = `Error: ${message}`;
        this.setAnimationLoading(false);
    }

    getTextInput() {
        return this.elements[ELEMENTS.TEXT_INPUT] ? this.elements[ELEMENTS.TEXT_INPUT].value.trim() : "";
    }

    getCurrentMode() {
        return this.currentMode;
    }
}