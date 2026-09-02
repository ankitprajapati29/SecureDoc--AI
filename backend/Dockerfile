FROM python:3.11-slim-bookworm

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application
COPY main.py .

# Render provides the PORT environment variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
