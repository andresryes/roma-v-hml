FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    build-essential \
    libssl-dev \
    libeigen3-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app source, data, and entrypoint
COPY app ./app
COPY data ./data
COPY entrypoint.sh ./entrypoint.sh

# Create output directory and ensure entrypoint is executable
RUN mkdir -p output && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
