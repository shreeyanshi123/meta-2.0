#!/usr/bin/env bash
# Run the Tribunal server locally with auto-reload for development.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "🏛️  Starting AI Agent Oversight Tribunal (dev mode)"
echo "   → http://localhost:8000"
echo "   → http://localhost:8000/docs  (Swagger)"
echo "   → http://localhost:8000/health"
echo ""

exec .venv/bin/uvicorn tribunal.server:app \
    --host "${TRIBUNAL_HOST:-127.0.0.1}" \
    --port "${TRIBUNAL_PORT:-8000}" \
    --reload \
    --reload-dir src
