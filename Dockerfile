# ── Stage 1: Build dashboard ──────────────────────────────────────
FROM node:20-alpine AS dashboard-builder

WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm ci --prefer-offline 2>/dev/null || npm install
COPY dashboard/ .
RUN npm run build

# ── Stage 2: Python runtime ──────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python packages — install in dependency order
COPY shared/ shared/
COPY client/ client/
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ./shared && \
    pip install --no-cache-dir ./client && \
    pip install --no-cache-dir .

# Copy remaining files
COPY openenv.yaml .
COPY scripts/ scripts/
COPY assets/ assets/

# Copy built dashboard from stage 1
COPY --from=dashboard-builder /dashboard/dist dashboard/dist

# HF Spaces uses port 7860
EXPOSE 7860

ENV TRIBUNAL_HOST=0.0.0.0
ENV TRIBUNAL_PORT=7860
ENV TRIBUNAL_SEED=42
ENV TRIBUNAL_EPISODES=5
ENV TRIBUNAL_FAILURE_RATE=0.6

CMD ["uvicorn", "tribunal.server:app", "--host", "0.0.0.0", "--port", "7860"]
