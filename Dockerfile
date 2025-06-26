# Stage 1: Build React frontend
FROM node:18 AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend with FastAPI
FROM python:3.10-slim

WORKDIR /app

# Install system packages required to build native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
#         libdbus-1-dev \
        libglib2.0-dev \
        libffi-dev \
        python3-dev \
        pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and source code
COPY api_web/ ./api_web/
COPY gcmswine/ ./gcmswine/
COPY scripts/ ./scripts/
# COPY datasets/ ./datasets/
COPY config.yaml ./
COPY setup.py ./

# Copy built frontend into static folder
COPY --from=frontend-build /frontend/build ./frontend-build

RUN mkdir -p /app/static

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api_web.main:app", "--host", "0.0.0.0", "--port", "8000"]