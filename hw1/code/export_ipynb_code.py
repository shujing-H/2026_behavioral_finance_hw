#!/usr/bin/env python3
"""
Export code cells from a Jupyter notebook (.ipynb) into a .py file.

We avoid nbconvert/jupyter dependencies by reading the notebook JSON directly.
The output is intended for report appendices (readable, with cell markers).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _cell_source_text(cell: Dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return ""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ipynb_path = repo_root / "hw1" / "code" / "hw1.ipynb"
    out_path = repo_root / "hw1" / "code" / "hw1_appendix.py"

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))

    lines = []
    lines.append('"""')
    lines.append("Auto-generated from hw1/code/hw1.ipynb.")
    lines.append("")
    lines.append("This file is intended for the homework report appendix.")
    lines.append('"""')
    lines.append("")
    lines.append("# flake8: noqa")
    lines.append("")

    cell_no = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell_no += 1
        src = _cell_source_text(cell).rstrip()
        # Normalize a few unicode symbols that can break LaTeX monospace fonts in listings.
        src = (
            src.replace("τ", "tau")
            .replace("α", "alpha")
            .replace("β", "beta")
            .replace("−", "-")  # unicode minus
            .replace("–", "-")  # en dash
        )
        if not src:
            continue
        lines.append(f"# %% [cell {cell_no}]")
        lines.append(src)
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {cell_no} code cells to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

