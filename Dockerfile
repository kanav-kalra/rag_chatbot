# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Core Python environment settings (NO SSL overrides)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and update CA certificates
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade certifi to ensure latest CA certificates
RUN pip install --upgrade --no-cache-dir certifi

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/logs /app/data/vectorstores

# Expose ports
# FastAPI default port
EXPOSE 8000
# Streamlit default port
EXPOSE 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.main", "--app", "both"]
