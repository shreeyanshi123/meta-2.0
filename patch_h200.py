import os
import sys

target_file = "/Users/hemantaggarwal/Documents/Meta-Pytorch-huggingFace-round2/training/train_grpo.py"

if not os.path.exists(target_file):
    print(f"File not found: {target_file}")
    sys.exit(1)

with open(target_file, "r") as f:
    content = f.read()

# Apply H200 Optimizations
# 1. Turn off 4-bit, use bf16 natively and Flash Attention 2
content = content.replace(
    "load_in_4bit=True, dtype=None",
    "load_in_4bit=False, dtype=torch.bfloat16, attn_implementation=\"flash_attention_2\""
)

# 2. Increase generations and batch sizes
content = content.replace("num_generations=4", "num_generations=8")
content = content.replace("per_device_train_batch_size=2", "per_device_train_batch_size=16")
content = content.replace("gradient_accumulation_steps=8", "gradient_accumulation_steps=2")

with open(target_file, "w") as f:
    f.write(content)

print("✅ Successfully updated train_grpo.py for H200 optimizations!")
