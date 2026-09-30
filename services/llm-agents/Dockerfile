FROM python:3.12-slim
 
WORKDIR /app
 
# Minimal system dependencies.
# libgomp1: OpenMP runtime required by the lightgbm manylinux wheel ("model" decision-maker).
# Without it the package installs but `import lightgbm` fails.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt
 
# Source code
COPY . .

 