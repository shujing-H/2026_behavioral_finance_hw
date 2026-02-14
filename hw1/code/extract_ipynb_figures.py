#!/usr/bin/env python3
"""
Extract embedded PNG figures from a Jupyter notebook (.ipynb).

This homework notebook stores matplotlib figures inside cell outputs as base64
under output.data["image/png"]. We decode and write them to disk with
deterministic names when we can infer the figure from the cell source.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _cell_source_text(cell: Dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return ""


def _iter_png_outputs(nb: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    """
    Returns list of (cell_index, output_index, png_b64).
    """
    out: List[Tuple[int, int, str]] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        for oi, o in enumerate(cell.get("outputs", []) or []):
            data = o.get("data", {}) if isinstance(o, dict) else {}
            png = data.get("image/png")
            if isinstance(png, str) and png.strip():
                out.append((ci, oi, png))
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ipynb_path = repo_root / "hw1" / "code" / "hw1.ipynb"
    assets_dir = repo_root / "hw1" / "docs" / "hw1_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))

    png_outputs = _iter_png_outputs(nb)
    if not png_outputs:
        raise RuntimeError(f"No embedded PNG outputs found in {ipynb_path}")

    # Heuristics for deterministic naming based on cell source.
    # We assign names per-cell; if multiple images exist in a single cell, we
    # use an ordered list of names.
    per_cell_names: Dict[int, List[str]] = {}
    for ci, cell in enumerate(nb.get("cells", [])):
        src = _cell_source_text(cell)
        if "plot_decile_portfolios" in src and "mom_12_1 (ew)" in src:
            per_cell_names[ci] = [
                "q1_cumret_ew.png",
                "q1_cumret_vw.png",
                "q1_cumret_rw.png",
            ]
        elif "# plot three long-short portfolios" in src or "plot three long-short portfolios" in src:
            per_cell_names[ci] = ["q1_longshort_timeseries.png"]
        elif "Plot FF alpha and t-statistics as a function of horizon" in src:
            per_cell_names[ci] = ["q4_horizon_alpha_tstat.png"]

    manifest: Dict[str, Any] = {
        "notebook": str(ipynb_path.relative_to(repo_root)),
        "assets_dir": str(assets_dir.relative_to(repo_root)),
        "extracted": [],
    }

    # Track how many PNGs we have written per cell to select names.
    written_per_cell: Dict[int, int] = {}
    generic_idx = 1

    for ci, oi, png_b64 in png_outputs:
        names = per_cell_names.get(ci)
        use_i = written_per_cell.get(ci, 0)
        if names and use_i < len(names):
            fname = names[use_i]
        else:
            fname = f"fig_{generic_idx:02d}_cell{ci}_out{oi}.png"
            generic_idx += 1

        written_per_cell[ci] = use_i + 1

        out_path = assets_dir / fname
        png_bytes = base64.b64decode(png_b64.encode("ascii"))
        out_path.write_bytes(png_bytes)

        manifest["extracted"].append(
            {
                "file": str(out_path.relative_to(repo_root)),
                "cell_index": ci,
                "output_index": oi,
            }
        )

    (assets_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Extracted {len(manifest['extracted'])} PNG(s) to {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

