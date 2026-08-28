# tools/ – build & validation harness

Every notebook in this repo is **generated** from a small builder module so the repo stays reproducible and reviewable.

## Layout
- `builders/_base.py` – shared notebook writer (`nbformat`)
- `builders/ch01.py` … `builders/ch12.py` – the cell-by-cell content of each chapter
- `build_all.py` – rebuild all notebooks: `python tools/build_all.py`
- `run_all.py` – execute every notebook end-to-end: `python tools/run_all.py`
- `validate.py` – fast sanity check (JSON well-formed, no syntax errors)

## Why build notebooks from code?
You can regenerate any notebook after tweaking its builder, inspect diffs easily,
and CI can re-run all of them on a clean environment.