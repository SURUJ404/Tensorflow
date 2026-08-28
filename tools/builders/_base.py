"""Shared helper to build .ipynb notebooks from Python cell definitions."""
from __future__ import annotations

import json
import nbformat as nbf
from pathlib import Path

PY3 = {"display_name": "Python 3", "metadata": {"language_info": {"name": "python"}}, "name": "python3", "language": "python"}


def make_notebook(title: str, cells: list):
    """Create a notebook object from a list of (kind, source) tuples.

    kind is "md" (markdown) or "code".
    """
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": PY3, "language_info": {"name": "python"}}
    out_cells = []
    first_md_done = False
    for kind, source in cells:
        src = source.strip("\n")
        if kind == "md":
            c = nbf.v4.new_markdown_cell(src)
        elif kind == "code":
            c = nbf.v4.new_code_cell(src)
            c.metadata = {"trusted": False}
        else:
            raise ValueError(kind)
        out_cells.append(c)
    # Not all cells need an empty leading markdown, but ensure title doc.
    nb.cells = out_cells
    return nb


def write_notebook(path: str | Path | "Path", cells: list, title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = make_notebook(title, cells)
    text = nbf.writes(nb, indent=1)
    path.write_text(text, encoding="utf-8")
    return path