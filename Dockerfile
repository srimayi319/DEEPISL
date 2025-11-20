# 1. Use a lightweight Python 3.10 image
# 'slim' version saves space on the cloud server
FROM python:3.10-slim

# 2. Install System Dependencies (CRITICAL for OpenCV & MediaPipe)
# Linux servers don't have video drivers by default. 
# We must install 'libgl1' and 'libglib2.0' manually or the app will crash.
# NEW (Fixed)
# Added 'ffmpeg' to the list of installed packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install Python Libraries
# We copy requirements.txt first to use Docker's cache (makes re-deploying faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download the SpaCy Language Model
# This is required for your Grammar Engine logic
RUN python -m spacy download en_core_web_sm

# 6. Copy the rest of your application code
COPY . .

# 7. Configure the Port
# Render provides the port in an environment variable, but we set a default
ENV PORT=5000
EXPOSE 5000

# 8. Start the Application
# We use Gunicorn with 100 threads to handle WebSocket connections smoothly.
# Correct command for simple-websocket
CMD gunicorn -w 1 --threads 100 --bind 0.0.0.0:$PORT app:app