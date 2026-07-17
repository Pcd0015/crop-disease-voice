# Crop Disease Voice Assistant — container image for Render.com (Docker runtime)
FROM python:3.11-slim

WORKDIR /app

# System libraries needed by audio (gTTS/soundfile) and image (Pillow/opencv-headless) deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching — only reinstalls when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (respects .dockerignore below)
COPY . .

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Create a non-root user and hand it ownership of /app — without this,
# files copied in as root can't have new subdirectories (like storage/)
# created inside them at runtime by the app process
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/storage && \
    chown -R appuser:appuser /app
USER appuser

CMD streamlit run streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0
