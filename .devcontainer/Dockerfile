# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    # PostgreSQL client
    postgresql-client \
    libpq-dev \
    # Python development headers
    python3-dev \
    # For data processing and ML
    libgomp1 \
    # Additional useful tools
    vim \
    nano \
    htop \
    tmux \
    # For Jupyter kernel
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Install pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipython \
    black \
    isort \
    pytest \
    pytest-cov \
    pylint \
    flake8 \
    mypy

# Copy requirements (will be updated by postStartCommand if needed)
COPY requirements.txt .

# Pre-install PyTorch separately (large package, benefits from being cached)
RUN pip install --no-cache-dir \
    torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install core scientific packages
RUN pip install --no-cache-dir \
    numpy==2.4.4 \
    pandas==3.0.2 \
    scikit-learn==1.8.0 \
    scipy==1.17.1 \
    matplotlib==3.10.9 \
    seaborn==0.13.2

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /workspace/logs \
    && mkdir -p /workspace/models \
    && mkdir -p /workspace/data

# Set up git safe directory
RUN git config --global --add safe.directory /workspace

# Expose ports
EXPOSE 8501 5432 5433

# Default command
CMD ["/bin/bash"]
