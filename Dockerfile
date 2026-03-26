# Build:
#   docker build -t fall-cpp:latest .
#
# Run (code baked in; start.sh compiles then launches all 3 processes):
#   docker run --gpus all --rm --env-file .env fall-cpp:latest
#
# Or mount code for live dev (start.sh rebuilds on change):
#   docker run --gpus all --rm -v $(pwd):/home/app --env-file .env fall-cpp:latest

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# System dependencies
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libopencv-dev \
    librdkafka-dev \
    libssl-dev \
    libzmq3-dev \
    ninja-build \
    pkg-config \
    python3 \
    tar \
    unzip \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# LibTorch — CUDA 11.8 build (matches T4 / sm_75)
# ==============================================================================
RUN wget -q "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu118.zip" \
         -O /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

ENV LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH:-}

# ==============================================================================
# vcpkg
# ==============================================================================
ENV VCPKG_DEFAULT_TRIPLET=x64-linux-dynamic
RUN git clone --depth 1 --branch 2024.11.16 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    || git clone --depth 1 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh

# ==============================================================================
# Application
# ==============================================================================
WORKDIR /home/app

# T4 GPU = compute capability 7.5 — pre-set so start.sh skips nvidia-smi detection
ENV TORCH_CUDA_ARCH_LIST=7.5

# Copy requirements first for better layer caching (vcpkg deps won't reinstall if vcpkg.json unchanged)
COPY vcpkg.json .
COPY CMakeLists.txt .
COPY app/ app/

COPY start.sh .
RUN chmod +x ./start.sh

CMD ["bash", "./start.sh"]
