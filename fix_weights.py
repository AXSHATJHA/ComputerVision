import json

with open("implementation.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # fix checkpoint extension
        if "'./checkpoints/my_checkpoint'" in source:
            source = source.replace("'./checkpoints/my_checkpoint'", "'./checkpoints/my_checkpoint.weights.h5'")
            cell["source"] = source.splitlines(keepends=True)

with open("implementation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
