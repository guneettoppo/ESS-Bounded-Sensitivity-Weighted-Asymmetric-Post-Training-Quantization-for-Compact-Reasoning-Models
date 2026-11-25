"""
Notebook scaffolding.

Why generate notebooks from Python instead of hand-editing .ipynb JSON
----------------------------------------------------------------------
Every notebook must be self-contained (Kaggle gives each notebook a fresh VM, so an
import from a sibling notebook's /kaggle/working does not work unless the user wires
up a dataset). Self-contained means the ~2.5k-line library is embedded at the top of
every notebook -- so it has to be *generated* from the canonical src/ tree, or the ten
copies will drift apart within a week.

Layout of every notebook:
    cell 0   library + configs (generated, identical everywhere)
    cell 1   versions / GPU
    cell 2   prerequisites check (fails loudly, names the notebook to run first)
    cell 3+  the actual stage
    last     artifact manifest + handoff to the next notebook
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "ebswgptaq")
CONFIGS = os.path.join(ROOT, "configs")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def library_cell() -> str:
    """One code cell that reconstructs the package and the config files on disk."""
    modules = {}
    for name in sorted(os.listdir(SRC)):
        if name.endswith(".py"):
            modules[name] = _read(os.path.join(SRC, name))
    configs = {}
    for name in sorted(os.listdir(CONFIGS)):
        if name.endswith(".yaml"):
            configs[name] = _read(os.path.join(CONFIGS, name))

    parts = [
        "# =====================================================================================",
        "# CELL 0 -- LIBRARY AND CONFIGS (generated; identical in every notebook)",
        "#",
        "# Source of truth: D:\\BTP code\\src\\ebswgptaq  (edit there, then rebuild the",
        "# notebooks with scripts/build_notebooks.py). Do not edit this cell by hand.",
        "# =====================================================================================",
        "import os, sys, json",
        "",
        '_ROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()',
        'PKG_DIR = os.path.join(_ROOT, "ebswgptaq")',
        'CONFIG_DIR = os.path.join(_ROOT, "configs")',
        "os.makedirs(PKG_DIR, exist_ok=True)",
        "os.makedirs(CONFIG_DIR, exist_ok=True)",
        "",
        "MODULES = {}",
    ]
    for name, src in modules.items():
        parts.append(f"MODULES[{name!r}] = r'''{src}'''")
    parts.append("")
    parts.append("CONFIGS = {}")
    for name, src in configs.items():
        parts.append(f"CONFIGS[{name!r}] = r'''{src}'''")
    parts += [
        "",
        "for _name, _src in MODULES.items():",
        '    with open(os.path.join(PKG_DIR, _name), "w", encoding="utf-8", newline="\\n") as _fh:',
        "        _fh.write(_src)",
        "for _name, _src in CONFIGS.items():",
        '    with open(os.path.join(CONFIG_DIR, _name), "w", encoding="utf-8", newline="\\n") as _fh:',
        "        _fh.write(_src)",
        "",
        "if _ROOT not in sys.path:",
        "    sys.path.insert(0, _ROOT)",
        "import ebswgptaq",
        "print(f'library ready: v{ebswgptaq.__version__} from {ebswgptaq.__file__}')",
        "print(f'{len(MODULES)} modules, {len(CONFIGS)} configs written to {_ROOT}')",
    ]
    return "\n".join(parts)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _split(text)}


def _split(text: str) -> list:
    text = text.strip("\n")
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    out.append(lines[-1])
    return out


KERNELSPEC = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}


def write_notebook(path: str, cells: list, title: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {**KERNELSPEC, "title": title, "accelerator": "GPU",
                     "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                "isGpuEnabled": True}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path


HEADER_CELL = """\
# Prerequisites check
# Fails immediately, naming the notebook that produces the missing artifact.
from ebswgptaq import logging_utils as lu

print("artifact root:", lu.write_root())
print("search roots :", lu.search_roots())
"""

EPILOGUE_TEMPLATE = """\
# ---------------------------------------------------------------------------------
# Artifact manifest -- what this notebook produced
# ---------------------------------------------------------------------------------
from ebswgptaq import logging_utils as lu
import os

written = __WRITTEN__
print("Artifacts written by this notebook:")
for rel in written:
    path = lu.out(rel)
    size = os.path.getsize(path) if os.path.exists(path) else -1
    print("  %-52s %8.2f MB" % (rel, size / 1e6))

print()
print("HANDOFF")
print("-" * 78)
print(__HANDOFF__)
print()
print("Next steps:")
print("  1. Save this notebook's version (Save Version -> Save & Run All) so the")
print("     artifacts above become that version's output.")
print("  2. In the next notebook: Add Input -> Notebook Output -> this notebook.")
print("     Its files then appear read-only under /kaggle/input/<slug>/artifacts/")
print("     and logging_utils.find() picks them up automatically.")
print("  3. Alternative: run lu.zip_artifacts('artifacts.zip'), download it, create a")
print("     private Kaggle dataset from the zip, and attach that instead.")
"""


def epilogue(written: list, handoff: str) -> str:
    """Build the closing cell without str.format (the template contains braces)."""
    text = EPILOGUE_TEMPLATE
    text = text.replace("__WRITTEN__", repr(list(written)))
    text = text.replace("__HANDOFF__", repr(handoff))
    return text
