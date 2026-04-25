import json

path = "/Users/hemantaggarwal/Documents/meta-2.0/notebooks/train_grpo.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        for i, line in enumerate(cell.get("source", [])):
            if 'load_in_4bit=False, dtype=torch.bfloat16, attn_implementation="flash_attention_2", \n' in line:
                cell["source"][i] = line.replace(', \n', ')\n')
            if 'load_in_4bit=False, dtype=torch.bfloat16, attn_implementation="flash_attention_2",\n' in line:
                cell["source"][i] = line.replace(',\n', ')\n')

with open(path, "w") as f:
    json.dump(nb, f, indent=1)

print("Fixed notebook syntax error.")
