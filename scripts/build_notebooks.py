"""
Build the ten Kaggle notebooks from the canonical library.

    python scripts/build_notebooks.py            # write notebooks/*.ipynb
    python scripts/build_notebooks.py --check    # also compile every code cell

Every notebook is self-contained: cell 0 reconstructs the `ebswgptaq` package and the
config files on disk, so a notebook works on a fresh Kaggle VM with nothing attached
except the artifacts of the previous notebook.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from nb_common import code, write_notebook, library_cell  # noqa: E402
import nb_00_04 as A  # noqa: E402
import nb_05_09 as B  # noqa: E402

NOTEBOOKS = [
    ("00_environment_check", A.NB00_TITLE, A.NB00),
    ("01_generate_traces", A.NB01_TITLE, A.NB01),
    ("02_uniform_baseline", A.NB02_TITLE, A.NB02),
    ("03_build_probe", A.NB03_TITLE, A.NB03),
    ("04_compute_jsd", A.NB04_TITLE, A.NB04),
    ("05_analyze_geometry", B.NB05_TITLE, B.NB05),
    ("06_run_weighted_variants", B.NB06_TITLE, B.NB06),
    ("07_evaluate_proxy", B.NB07_TITLE, B.NB07),
    ("08_probe_swap", B.NB08_TITLE, B.NB08),
    ("09_downstream_pilot", B.NB09_TITLE, B.NB09),
]

OUT_DIR = os.path.join(ROOT, "notebooks")


def assemble(cells: list) -> list:
    """Title markdown, then the generated library cell, then the stage cells."""
    return [cells[0], code(library_cell())] + list(cells[1:])


def compile_cell(src: str) -> str:
    """Return '' if the cell compiles, else the error message."""
    # Strip IPython magics and shell escapes so the cell can be parsed as Python.
    cleaned = []
    for line in src.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%", "%%")):
            indent = line[: len(line) - len(stripped)]
            cleaned.append(indent + "pass")
        else:
            cleaned.append(line)
    try:
        compile("\n".join(cleaned), "<cell>", "exec")
        return ""
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"


def main(check_only: bool = False) -> int:
    failures = []
    for filename, title, cells in NOTEBOOKS:
        nb_cells = assemble(cells)
        path = os.path.join(OUT_DIR, f"{filename}.ipynb")
        write_notebook(path, nb_cells, title=title)

        n_md = sum(1 for c in nb_cells if c["cell_type"] == "markdown")
        n_code = sum(1 for c in nb_cells if c["cell_type"] == "code")
        size = os.path.getsize(path) / 1024
        print(f"  {filename+'.ipynb':34s} {n_md:2d} md  {n_code:2d} code  {size:7.1f} KB")

        if check_only:
            for i, cell in enumerate(nb_cells):
                if cell["cell_type"] != "code":
                    continue
                err = compile_cell("".join(cell["source"]))
                if err:
                    failures.append((filename, i, err))

    if check_only:
        print()
        if failures:
            for f, i, err in failures:
                print(f"COMPILE FAIL {f} cell {i}: {err}")
            return 1
        print("all code cells compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
