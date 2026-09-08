# Tiny GPT

This project builds a small decoder-only GPT-like language model from scratch and progressively develops five model versions.

## Docker Workflow & Development Guide

This project is built around a **reusable Docker environment**. The core philosophy is to treat the **source code** and the **execution environment** as two completely separate things.

- **SOURCE CODE (GitHub):** The Python scripts, model configurations, and tests.
- **ENVIRONMENT (Docker Image):** Python, PyTorch, CUDA, and system dependencies.

You DO NOT need to rebuild the Docker image every time you edit a Python file.

### 1. Build the Docker Image (One-Time Setup)
Build the unified Docker image that supports both CPU and NVIDIA GPU workloads.
```bash
docker build -f docker/Dockerfile -t tiny-gpt:test .
```
> **Note:** The image is based on an official PyTorch container and is roughly ~3.6GB in size to include necessary CUDA libraries natively.

### 2. Standard Development (Bind Mounts)
To develop locally, you bind-mount your source code into the container. This means any code changes you make on your host machine are instantly reflected inside the container.

**CPU Mode:**
```bash
docker run -v "$(pwd):/app" --rm tiny-gpt:test python scripts/check_device.py
```

**NVIDIA GPU Mode (requires NVIDIA GPU, driver, and Container Toolkit):**
```bash
docker run --gpus all -v "$(pwd):/app" --rm tiny-gpt:test python scripts/validate_cuda.py
```

### 3. Running with Project Data and Checkpoints
Artifacts like datasets and checkpoints are not baked into the Docker image. Because you are bind-mounting the entire project directory (`-v "$(pwd):/app"`), the container naturally has read/write access to `data/` and `checkpoints/` exactly where the scripts expect them.

### 4. Updating Code vs. Updating Environment
If a friend makes changes to the model architecture:
1. **Pull the code:** `git pull`
2. **Run the container:** `docker run -v "$(pwd):/app" ...`
*No Docker image rebuild is required!*

**When IS a Docker rebuild required?**
Only rebuild the image if the environment dependencies change (e.g., adding a new pip package to `requirements.txt`).

### 5. Sharing the Environment (`docker save` / `docker load`)
If you want to share the exact environment without forcing a friend to download gigabytes of PyTorch layers, you can export and import the image:

```bash
# Export the environment
docker save tiny-gpt:test -o tiny-gpt-env.tar

# Your friend imports the environment
docker load -i tiny-gpt-env.tar
```
> **Important:** This only transfers the *environment*. Your friend must still `git clone` the repository to get the actual *source code*.

### 6. NVIDIA GPU Prerequisites
To use the `--gpus all` flag, the host machine must have:
1. A physical NVIDIA GPU.
2. An appropriate NVIDIA display driver installed on the host OS.
3. The NVIDIA Container Toolkit installed (or Docker Desktop with WSL2 GPU support enabled).
The Docker image itself provides PyTorch and the CUDA runtime, but it cannot provide the hardware or the low-level host driver.
