# Notebooks — GRPO Judge Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USER/meta-2.0/blob/main/notebooks/train_grpo.ipynb)

## `train_grpo.ipynb`

End-to-end GRPO training notebook for the AI Agent Oversight Tribunal Judge.

### What it does

1. Installs deps (Unsloth, TRL ≥ 0.11, tribunal packages)
2. Starts the Tribunal env server inside Colab (`uvicorn` background process)
3. Loads `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` with LoRA (r=16, α=32)
4. Defines a 2-few-shot Judge prompt template
5. Creates **6 independent reward functions** (one per component) for TRL
6. Builds a 2000-prompt dataset from the env
7. Trains with `GRPOTrainer` for 400 steps
8. Evaluates trained vs baseline over 200 rounds
9. Saves LoRA adapter (no naive merge!) and plots

### Hardware Requirements

| Runtime | GPU  | VRAM | Est. Time |
|---------|------|------|-----------|
| Colab Free | T4   | 15 GB | ~3–4 h   |
| Colab Pro  | A100 | 40 GB | ~45 min  |
| Local      | Any ≥16 GB | — | Varies |

### Memory Fallback

If `Qwen2.5-1.5B-Instruct-bnb-4bit` OOMs on T4, the notebook automatically falls back to `Qwen2.5-0.5B-Instruct` (4-bit, ~2 GB VRAM).

### Key Config

```
num_generations=4, batch=2, grad_accum=8
lr=5e-6, beta=0.02, max_steps=400
max_prompt=2048, max_completion=384
LoRA: r=16, alpha=32, all linear modules
```

### Outputs

| File | Description |
|------|-------------|
| `assets/eval_summary.json` | Per-component metrics (trained model) |
| `assets/reward_curve.png` | Total reward over training steps |
| `assets/per_component_rewards.png` | 6-component bar chart |
| `grpo_tribunal_judge/lora_adapter/` | Saved LoRA weights |

### Loading the Trained Judge

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "YOUR_USERNAME/tribunal-judge-qwen2.5-lora",
    max_seq_length=2432,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### WandB

Training logs to project `tribunal-judge`. Set `WANDB_API_KEY` env var or run `wandb login` before training.
