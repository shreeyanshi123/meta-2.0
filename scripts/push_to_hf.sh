#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# push_to_hf.sh — Push tribunal env to HF Spaces + LoRA adapter to Hub
#
# Usage:
#   HF_USER=your-username ./scripts/push_to_hf.sh
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

HF_USER="${HF_USER:?Set HF_USER to your HuggingFace username}"
SPACE_NAME="${SPACE_NAME:-tribunal-env}"
ADAPTER_NAME="${ADAPTER_NAME:-tribunal-judge-lora}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "  🏛️  Pushing to HuggingFace"
echo "  Space:   ${HF_USER}/${SPACE_NAME}"
echo "  Adapter: ${HF_USER}/${ADAPTER_NAME}"
echo "════════════════════════════════════════════════════════════════"

# ── 1. Push Space (Docker SDK) ────────────────────────────────────
echo ""
echo "▶ Step 1: Pushing HF Space..."

# Create Space repo if it doesn't exist
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        '${HF_USER}/${SPACE_NAME}',
        repo_type='space',
        space_sdk='docker',
        exist_ok=True,
    )
    print('  ✅ Space repo ready')
except Exception as e:
    print(f'  ⚠️  {e}')
"

# Upload entire repo to Space
python3 -c "
from huggingface_hub import HfApi
api = HfApi()

# Files to include in the Space
api.upload_folder(
    folder_path='${REPO_ROOT}',
    repo_id='${HF_USER}/${SPACE_NAME}',
    repo_type='space',
    ignore_patterns=[
        '.git/*',
        '.github/*',
        'runs/*',
        'wandb/*',
        'grpo_tribunal_judge/*',
        'node_modules/*',
        '*.pyc',
        '__pycache__/*',
        '.venv/*',
        '.eggs/*',
        '*.egg-info/*',
        '.pytest_cache/*',
    ],
)
print('  ✅ Space uploaded successfully')
print(f'  🌐 https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}')
"

# ── 2. Push LoRA adapter ──────────────────────────────────────────
ADAPTER_DIR="${REPO_ROOT}/grpo_tribunal_judge/lora_adapter"

if [ -d "${ADAPTER_DIR}" ]; then
    echo ""
    echo "▶ Step 2: Pushing LoRA adapter..."

    python3 -c "
from huggingface_hub import HfApi
api = HfApi()

# Create model repo
api.create_repo(
    '${HF_USER}/${ADAPTER_NAME}',
    repo_type='model',
    exist_ok=True,
)

# Upload adapter files
api.upload_folder(
    folder_path='${ADAPTER_DIR}',
    repo_id='${HF_USER}/${ADAPTER_NAME}',
    repo_type='model',
)
print('  ✅ LoRA adapter uploaded')
print(f'  🧠 https://huggingface.co/${HF_USER}/${ADAPTER_NAME}')
"
else
    echo ""
    echo "▶ Step 2: Skipping LoRA adapter (${ADAPTER_DIR} not found)"
    echo "  Run the training notebook first to generate the adapter."
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Push complete!"
echo ""
echo "  Space:     https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
if [ -d "${ADAPTER_DIR}" ]; then
    echo "  Adapter:   https://huggingface.co/${HF_USER}/${ADAPTER_NAME}"
fi
echo "════════════════════════════════════════════════════════════════"
