FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy remaining files
COPY openenv.yaml .
COPY dashboard/ dashboard/
COPY scripts/ scripts/

EXPOSE 8000

ENV TRIBUNAL_HOST=0.0.0.0
ENV TRIBUNAL_PORT=8000
ENV TRIBUNAL_SEED=42
ENV TRIBUNAL_EPISODES=5
ENV TRIBUNAL_FAILURE_RATE=0.6

CMD ["uvicorn", "tribunal.server:app", "--host", "0.0.0.0", "--port", "8000"]
