FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pandas/numpy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir pandas==2.0.3
RUN pip install --no-cache-dir xgboost==1.7.6
RUN pip install --no-cache-dir scikit-learn==1.3.0
RUN pip install --no-cache-dir fastapi==0.111.0
RUN pip install --no-cache-dir 'uvicorn[standard]==0.30.1'
RUN pip install --no-cache-dir websockets==12.0
RUN pip install --no-cache-dir websocket-client==1.8.0

# Copy your application code
COPY . .

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]