import json

path = "/teamspace/studios/this_studio/meta-2.0/notebooks/train_grpo.ipynb"
try:
    with open(path, "r") as f:
        nb = json.load(f)
except FileNotFoundError:
    path = "notebooks/train_grpo.ipynb"
    with open(path, "r") as f:
        nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell["source"])
        
        # Force bfloat16 in model loading
        if "FastLanguageModel.from_pretrained(" in source:
            source = source.replace("dtype=None", "dtype=torch.bfloat16")
            # If the user already added it but it's wrong, we can do a regex or just replace
            cell["source"] = [line + "\n" for line in source.split("\n")]
            
        # Ensure GRPOConfig uses bf16=True, fp16=False and WANDB_MODE=offline
        if "GRPOConfig(" in source:
            source = source.replace("bf16=torch.cuda.is_bf16_supported()", "bf16=True")
            source = source.replace("fp16=not torch.cuda.is_bf16_supported()", "fp16=False")
            source = source.replace("max_steps=50", "max_steps=30")
            source = source.replace("max_completion_length=384", "max_completion_length=256")
            source = source.replace("num_generations=4", "num_generations=2")
            source = source.replace("gradient_accumulation_steps=8", "gradient_accumulation_steps=4")
            source = source.replace("per_device_train_batch_size=2", "per_device_train_batch_size=1")
            
            cell["source"] = [line + "\n" for line in source.split("\n")[:-1]]
            
        if "wandb.init(" in source and "WANDB_MODE" not in source:
            source = 'import os\nos.environ["WANDB_MODE"] = "offline"\n' + source
            cell["source"] = [line + "\n" for line in source.split("\n")[:-1]]

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print("Successfully patched notebook with strict bfloat16 dtypes and fast settings!")
