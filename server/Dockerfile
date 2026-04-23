FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU-only first (smaller image, ~700MB instead of ~2GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skip torch/torchvision since already installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create persistent data directories
RUN mkdir -p /data/lancedb_db /data/images

# Environment variables for deployment
ENV LANCEDB_URI=/data/lancedb_db
ENV DATA_DIR=/data/images
ENV PORT=7860
ENV EMBED_MODEL=google/siglip-base-patch16-256
ENV EMBED_DIM=768
ENV BACKEND_URL=https://aariz-s-segp-siglip-search.hf.space
ENV HF_DATASET_REPO=aariz-s/segp-search-data

# Expose port (HF Spaces expects 7860)
EXPOSE 7860

# Start the server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
