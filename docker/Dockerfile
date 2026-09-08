# Base image with PyTorch and CUDA runtime already installed.
# This prevents Docker Desktop disk exhaustion from extracting massive PyPI wheels.
# It natively supports both CPU and NVIDIA GPU execution.
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install project dependencies
# (Torch is already installed in the base image, so pip will skip it or verify it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We explicitly DO NOT COPY the source code here.
# The source code should be volume-mounted at runtime to allow development
# without rebuilding the image: docker run -v "$(pwd):/app" ...
