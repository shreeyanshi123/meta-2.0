"""Fix train_grpo.ipynb for H200 GPU - clean rewrite of broken cells."""
import json

PATH = "/Users/hemantaggarwal/Documents/meta-2.0/notebooks/train_grpo.ipynb"

with open(PATH, "r") as f:
    nb = json.load(f)

# ── Fix Cell 0: Header markdown ──
nb["cells"][0]["source"] = [
    "# 🏛️ AI Agent Oversight Tribunal — GRPO Judge Training\n",
    "> **Target**: H200 (141 GB HBM3e, Hopper arch)  \n",
    "> **Base model**: `unsloth/Qwen2.5-1.5B-Instruct` (native bf16, no quantization)  \n",
    "> **Framework**: TRL GRPOTrainer + Unsloth  \n",
    "> **Hackathon**: Meta × HuggingFace OpenEnv India 2026\n"
]

# ── Fix Cell 1 (Install): add flash-attn ──
nb["cells"][2]["source"] = [
    "%%capture\n",
    "!pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
    "!pip install -q \"trl>=0.11\" transformers accelerate bitsandbytes datasets wandb\n",
    "!pip install -q openenv httpx rapidfuzz pydantic>=2 rich numpy\n",
    "!pip install -q flash-attn --no-build-isolation\n"
]

# ── Fix Cell 4: Model loading (the broken one) ──
# Find the cell with "Load Base Model"
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "FastLanguageModel.from_pretrained" in src and cell["cell_type"] == "code":
        nb["cells"][i]["source"] = [
            "import torch\n",
            "from unsloth import FastLanguageModel\n",
            "\n",
            "# H200: Use full-precision bf16 models — no 4-bit quantization needed\n",
            "PRIMARY = \"unsloth/Qwen2.5-1.5B-Instruct\"\n",
            "FALLBACK = \"unsloth/Qwen2.5-0.5B-Instruct\"\n",
            "MAX_SEQ = 2048 + 384\n",
            "\n",
            "try:\n",
            "    model, tokenizer = FastLanguageModel.from_pretrained(\n",
            "        model_name=PRIMARY,\n",
            "        max_seq_length=MAX_SEQ,\n",
            "        load_in_4bit=False,\n",
            "        dtype=torch.bfloat16,\n",
            "    )\n",
            "    MODEL_NAME = PRIMARY\n",
            "except Exception as e:\n",
            "    print(f\"⚠️  Primary failed ({e}), loading fallback...\")\n",
            "    model, tokenizer = FastLanguageModel.from_pretrained(\n",
            "        model_name=FALLBACK,\n",
            "        max_seq_length=MAX_SEQ,\n",
            "        load_in_4bit=False,\n",
            "        dtype=torch.bfloat16,\n",
            "    )\n",
            "    MODEL_NAME = FALLBACK\n",
            "\n",
            "print(f\"Loaded: {MODEL_NAME}\")\n",
            "\n",
            "# H200: Higher LoRA rank (32) since we have the VRAM headroom\n",
            "model = FastLanguageModel.get_peft_model(\n",
            "    model,\n",
            "    r=32, lora_alpha=64,\n",
            "    target_modules=[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\n",
            "                     \"gate_proj\",\"up_proj\",\"down_proj\"],\n",
            "    lora_dropout=0, bias=\"none\", use_gradient_checkpointing=\"unsloth\",\n",
            ")\n",
            "print(\"LoRA adapters attached\")\n"
        ]
        break

# ── Fix Cell 9: GRPOConfig for H200 ──
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "GRPOConfig(" in src and "training_args" in src and cell["cell_type"] == "code":
        nb["cells"][i]["source"] = [
            "from trl import GRPOConfig, GRPOTrainer\n",
            "\n",
            "# H200-optimized config: large batch, native bf16, no fp16 fallback\n",
            "training_args = GRPOConfig(\n",
            "    output_dir=\"./grpo_tribunal_judge\",\n",
            "    num_generations=8,\n",
            "    per_device_train_batch_size=16,\n",
            "    gradient_accumulation_steps=2,\n",
            "    learning_rate=5e-6,\n",
            "    max_prompt_length=2048,\n",
            "    max_completion_length=384,\n",
            "    beta=0.02,\n",
            "    num_train_epochs=1,\n",
            "    max_steps=50,\n",
            "    logging_steps=10,\n",
            "    save_steps=100,\n",
            "    warmup_steps=20,\n",
            "    bf16=True,\n",
            "    fp16=False,\n",
            "    optim=\"adamw_torch\",\n",
            "    report_to=\"wandb\",\n",
            "    run_name=\"tribunal-judge-grpo-h200\",\n",
            "    seed=42,\n",
            ")\n",
            "print(\"GRPOConfig ready ✅\")\n",
            "print(f\"  Steps: {training_args.max_steps}, LR: {training_args.learning_rate}\")\n",
            "print(f\"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}\")\n"
        ]
        break

# ── Fix Cell 12: Save adapter (has extra closing paren) ──
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "HUB_REPO" in src and "save_pretrained" in src and cell["cell_type"] == "code":
        nb["cells"][i]["source"] = [
            "# ⚠️  DO NOT merge + upcast — save adapters only\n",
            "HUB_REPO = \"shreeyanshi123NAME/tribunal-judge-qwen2.5-lora\"\n",
            "\n",
            "model.save_pretrained(\"./grpo_tribunal_judge/lora_adapter\")\n",
            "tokenizer.save_pretrained(\"./grpo_tribunal_judge/lora_adapter\")\n",
            "print(\"Saved LoRA adapter locally\")\n",
            "\n",
            "# Push to Hub (uncomment when ready)\n",
            "# model.push_to_hub(HUB_REPO, token=os.environ.get(\"HF_TOKEN\"))\n",
            "# tokenizer.push_to_hub(HUB_REPO, token=os.environ.get(\"HF_TOKEN\"))\n",
            "\n",
            "print(f\"To load the trained judge:\")\n",
            "print(f\"  model, tok = FastLanguageModel.from_pretrained('{HUB_REPO}', max_seq_length=2432, load_in_4bit=False, dtype=torch.bfloat16)\")\n"
        ]
        break

with open(PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("✅ Notebook fully fixed for H200!")
