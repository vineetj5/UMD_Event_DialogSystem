FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. System Dependencies
# libpq-dev is for psycopg2. build-essential/gcc/g++ are needed for compiling 
# dependencies like HDBScan (used by BERTopic) if wheels aren't available.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Install Torch CPU-only First (Caching Layer)
# We do this separately to avoid re-downloading 1GB+ if you only change a small package later.
RUN pip install --no-cache-dir torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install Requirements
COPY requirements.txt .

# CRITICAL CHANGE: 
# Removed "--no-deps" so pip installs sub-dependencies (fixes pytz/dateutil error).
# Added "--extra-index-url" so if any package tries to update torch/numpy, 
# it looks at the CPU channel instead of downloading massive GPU binaries.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copy Code
COPY loader.py .
COPY app.py .
# COPY scrape.py . # Uncomment if needed