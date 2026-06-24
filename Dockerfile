# Dockerfile

# ==========================================
# Stage 1: Build & Compile Dependencies
# ==========================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy structural files to leverage layer caching during updates
COPY pyproject.toml .

# Compile wheels into a local wheelhouse folder to isolate compilation bloat
RUN pip3 wheel --no-cache-dir --wheel-dir=/build/wheels .

# ==========================================
# Stage 2: Minimal Production Runtime
# ==========================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

# Core Infrastructure Protocol: Enforce absolute temporal agreement via UTC
ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull only pre-compiled application wheels from the builder layer
COPY --from=builder /build/wheels /app/wheels
RUN pip3 install --no-index --find-links=/app/wheels /app/wheels/*.whl \
    && rm -rf /app/wheels

# Copy loose flat-layout execution scripts and modules directly into working root
COPY environment/ /app/environment/
COPY features/ /app/features/
COPY models/ /app/models/
COPY xagusd_sandbox/ /app/xagusd_sandbox/

CMD ["python3.11", "models/lstm_ppo.py"]