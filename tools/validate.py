"""Validate every built notebook: JSON well-formed, code cells parse cleanly."""
import json
import os
import ast

files = []
for root, dirs, fs in os.walk("."):
    if ".git" in root or "builders" in root or "tools" in root:
        continue
    for fn in fs:
        if fn.endswith(".ipynb"):
            files.append(os.path.join(root, fn))
files = sorted(files)

bad = 0
for f in files:
    with open(f, encoding="utf-8") as fh:
        nb = json.load(fh)
    assert isinstance(nb, dict) and isinstance(nb.get("cells"), list), f"{f}: not a notebook"
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            try:
                ast.parse(src)
            except SyntaxError as e:
                bad += 1
                print(f"{f} cell {i} SYNTAX ERROR: {e}")
print(f"checked {len(files)} notebooks, {bad} syntax errors")