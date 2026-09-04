import json
import sys

with open("D:\\DunderCode\\kaggle-kaggriculture\\submissions\\kaggriculture-findings-from-zero-to-top-meta.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]

with open("D:\\DunderCode\\kaggle-kaggriculture\\submissions\\opponent_meta.py", "w", encoding="utf-8") as f:
    for cell in code_cells:
        if isinstance(cell, list):
            f.write("".join(cell))
        else:
            f.write(cell)
        f.write("\n\n")

print("Extracted to opponent_meta.py")
