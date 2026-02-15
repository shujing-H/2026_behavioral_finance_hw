## Course Homework Repository

This repository contains my coursework/homework for **230S**. It’s primarily organized by assignment (e.g., `hw1/`) and includes the code, writeups, and generated artifacts used to produce the submitted results.

Please feel free to **use this repo as a reference for structure and workflow** (project layout, analysis pipeline, LaTeX integration, etc.), but **do not copy/paste** code or writeup content for your own submissions—follow your course’s academic integrity policy.

## What’s inside

- **`hw1/`**: Homework 1 materials
  - **`hw1/code/`**: Analysis notebook and helper scripts
    - `hw1.ipynb`: Main analysis notebook
    - `generate_report_tables.py`: Produces LaTeX tables used by the report
    - `extract_ipynb_figures.py`: Extracts figures from the notebook (when applicable)
    - `export_ipynb_code.py`: Exports notebook code into a script-like format (when applicable)
    - `hw1_appendix.py`: Supporting/appendix code
  - **`hw1/docs/`**: Report sources and compiled PDF
    - `hw1_report.tex`: LaTeX report source
    - `hw1_report.pdf`: Compiled report (tracked here for convenience)
    - `hw1_report_tables/`: Generated `*.tex` tables included by the report
  - **`hw1/data/`**: Local datasets (typically not tracked)

- **`pyproject.toml` / `uv.lock`**: Python environment and pinned dependencies (Python >= 3.12)
- **`.mplconfig/`**: Matplotlib configuration for consistent plotting (when used)

## Setup (Python)

This repo is set up as a small Python project using `uv`.

```bash
uv sync
```

That creates/updates a local virtual environment (see `.venv/`) with the dependencies declared in `pyproject.toml`.

## Reproducing outputs (Homework 1)

Exact reproduction depends on having the same input datasets (often kept locally and ignored by git). With data in place, the typical workflow is:

```bash
# From the repo root:
uv run python hw1/code/generate_report_tables.py
```

To run the notebook, open `hw1/code/hw1.ipynb` in VS Code/Cursor and select the `.venv` Python interpreter/kernel.

Then compile the report in `hw1/docs/` using your LaTeX toolchain (or open `hw1/docs/hw1_report.pdf` if you just want to read it).

## Notes

- **Data & large artifacts**: `data/`, `*.csv`, and `*.parquet` are ignored by default (see `.gitignore`). Some PDFs are committed here for convenience even though PDFs are generally ignored.
- **Not a library**: This is coursework, not a reusable package. Expect code to be optimized for clarity and assignment requirements rather than generality.
