"""
Artifact paths, IO helpers, and the reproducibility manifest.

Kaggle reality this module solves
---------------------------------
Every Kaggle notebook runs on a fresh VM. `/kaggle/working` does not survive into
another notebook unless that notebook's output is attached as an input, in which
case it appears read-only under `/kaggle/input/<slug>/`.

So this module keeps two roots:
  write_root()  -> always writable  (/kaggle/working/artifacts, or $EBSW_ARTIFACT_ROOT)
  find(rel)     -> searches write_root() first, then every /kaggle/input/*/artifacts

`require(rel, produced_by=...)` raises an error that names the notebook you must run
first, which is the whole point: no silent empty-directory bugs.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------------------

WORK_ROOT = "/kaggle/working"


def write_root(subdir: str = "") -> str:
    """Writable artifact root. Created on demand."""
    base = os.environ.get("EBSW_ARTIFACT_ROOT")
    if not base:
        if os.path.isdir("/kaggle/working"):
            base = os.path.join(WORK_ROOT, "artifacts")
        else:
            # Local fallback so the library is importable off-Kaggle.
            base = os.path.join(os.getcwd(), "artifacts")
    path = os.path.join(base, subdir) if subdir else base
    os.makedirs(path, exist_ok=True)
    return path


def search_roots() -> List[str]:
    """All places `find` looks, in priority order."""
    roots = [write_root()]
    env = os.environ.get("EBSW_EXTRA_ROOTS")
    if env:
        roots.extend(p for p in env.split(os.pathsep) if p)
    roots.extend(sorted(glob.glob("/kaggle/input/*/artifacts")))
    return roots


def find(rel: str) -> Optional[str]:
    """Locate an artifact by relative path across all search roots."""
    for root in search_roots():
        candidate = os.path.join(root, rel)
        if os.path.exists(candidate):
            return candidate
    return None


def require(rel: str, produced_by: str) -> str:
    """Locate an artifact or fail loudly, naming the notebook that produces it."""
    path = find(rel)
    if path is None:
        raise FileNotFoundError(
            f"Missing required artifact: {rel}\n"
            f"  Produced by notebook: {produced_by}\n"
            f"  Searched: {search_roots()}\n"
            f"Fix: run {produced_by} first and save its version, then attach that\n"
            f"     version's output as an input to this notebook (Add Input ->\n"
            f"     Notebook Output), or set EBSW_ARTIFACT_ROOT to a directory\n"
            f"     containing it."
        )
    return path


def out(rel: str) -> str:
    """Path under the writable root; parent directories created."""
    path = os.path.join(write_root(), rel)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------


def save_json(obj: Any, rel: str) -> str:
    path = out(rel)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def load_json(rel_or_path: str, produced_by: str = "the notebook that creates it") -> Any:
    path = rel_or_path if os.path.isabs(rel_or_path) else require(rel_or_path, produced_by)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(obj: Any, rel: str) -> str:
    path = out(rel)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=_json_default) + "\n")
    return path


def read_jsonl(rel_or_path: str, produced_by: str = "the notebook that creates it") -> List[Dict]:
    path = rel_or_path if os.path.isabs(rel_or_path) else require(rel_or_path, produced_by)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], rel: str) -> str:
    path = out(rel)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=_json_default) + "\n")
    return path


def _json_default(obj: Any):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# --------------------------------------------------------------------------------------
# Run log (one JSONL line per stage, appended by every notebook)
# --------------------------------------------------------------------------------------


def log_event(stage: str, **fields) -> str:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage}
    record.update(fields)
    return append_jsonl(record, "logs/run_log.jsonl")


# --------------------------------------------------------------------------------------
# Reproducibility manifest (Milestone 0)
# --------------------------------------------------------------------------------------


def _pkg_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def environment_manifest(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything a reviewer needs to know to reproduce this run."""
    manifest: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _pkg_version(name)
            for name in ("torch", "transformers", "datasets", "tokenizers", "accelerate",
                         "numpy", "scipy", "safetensors", "sentencepiece", "yaml")
        },
    }
    try:
        import torch

        manifest["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "capability": [torch.cuda.get_device_capability(i)
                           for i in range(torch.cuda.device_count())],
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        }
    except Exception as exc:  # pragma: no cover
        manifest["torch"] = {"error": str(exc)}

    try:
        manifest["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        manifest["git"] = None

    if extra:
        manifest.update(extra)
    return manifest


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def gpu_memory_report() -> Dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda": True,
            "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
            "free_gb": round(free / 2**30, 3),
            "total_gb": round(total / 2**30, 3),
        }
    except Exception as exc:  # pragma: no cover
        return {"cuda": False, "error": str(exc)}


def banner(text: str, width: int = 88) -> None:
    print("=" * width)
    print(text)
    print("=" * width)


def section(text: str, width: int = 88) -> None:
    print(f"\n--- {text} " + "-" * max(0, width - len(text) - 5))


def zip_artifacts(zip_name: str, rel_paths: Optional[List[str]] = None) -> str:
    """
    Zip selected artifacts so they can be re-attached as a Kaggle dataset.

    This is the escape hatch when attaching notebook outputs directly is awkward:
    download the zip, create a private dataset from it, attach that.
    """
    import zipfile

    root = write_root()
    target = os.path.join(WORK_ROOT, zip_name) if os.path.isdir(WORK_ROOT) else zip_name
    paths = rel_paths
    if paths is None:
        paths = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                paths.append(os.path.relpath(full, root))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in paths:
            full = os.path.join(root, rel)
            if os.path.exists(full):
                zf.write(full, os.path.join("artifacts", rel))
    return target
