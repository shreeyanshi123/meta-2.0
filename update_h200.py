import json
import os

path = "notebooks/train_grpo.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        if "load_in_4bit=True" in source:
            source = source.replace("load_in_4bit=True", "load_in_4bit=False, dtype=torch.bfloat16, attn_implementation=\"flash_attention_2\"")
            source = source.replace("dtype=None)", "")
            
        if "num_generations=4" in source:
            source = source.replace("num_generations=4", "num_generations=8")
            
        if "per_device_train_batch_size=2" in source:
            source = source.replace("per_device_train_batch_size=2", "per_device_train_batch_size=16")
            
        if "gradient_accumulation_steps=8" in source:
            source = source.replace("gradient_accumulation_steps=8", "gradient_accumulation_steps=2")
            
        cell["source"] = [line + "\n" for line in source.split("\n")]
        # remove trailing newlines on the last element if needed, but split("\n") does this naturally
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    
print("Successfully updated notebooks/train_grpo.ipynb for H200!")
