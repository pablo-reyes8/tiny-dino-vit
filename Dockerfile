# ---------------------------------------------------------------
# DINO — Training container
# ---------------------------------------------------------------
# Build:  docker build -t dino-train .
# Run:    docker run --gpus all \
#           -v $(pwd)/data:/app/data \
#           -v $(pwd)/checkpoints:/app/checkpoints \
#           dino-train python scripts/train_dino.py --config configs/set_up.yaml
# ---------------------------------------------------------------

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (training)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Persistent volumes for data and checkpoints
VOLUME ["/app/data", "/app/checkpoints"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python"]
CMD ["scripts/train_dino.py", "--config", "configs/set_up.yaml"]
