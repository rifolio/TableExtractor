# Use the official slim Python 3.10 image
FROM python:3.10-slim

# Install system deps required for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgl1 libglib2.0-0 \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy in CPU-only requirements
COPY requirements.txt .

# Install Python deps, pulling the CPU-only PyTorch wheels
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# Copy application code
WORKDIR /app
COPY . /app

# force Python to flush stdout/stderr immediately
ENV PYTHONUNBUFFERED=1

# Default command: adjust to your entrypoint
CMD ["python", "main.py"]
