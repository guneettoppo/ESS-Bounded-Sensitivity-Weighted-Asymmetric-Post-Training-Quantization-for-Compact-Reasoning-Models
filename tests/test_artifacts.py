"""
Artifact-contract tests (no GPU, no model).

The notebook chain depends on one promise: if notebook N writes
`calib/calib_corpus.jsonl`, then notebook N+1 can find it and gets a clear error when
it cannot. A silent empty directory or a JSON round-trip that turns a tensor into a
string would waste GPU hours before anyone notices.

Run: EBSW_ARTIFACT_ROOT=<tmp> .venv/Scripts/python.exe tests/test_artifacts.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

TMP = tempfile.mkdtemp(prefix="ebsw_artifacts_")
os.environ["EBSW_ARTIFACT_ROOT"] = os.path.join(TMP, "artifacts")

from ebswgptaq import logging_utils as lu  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


import json

import torch

print("artifact root:", lu.write_root())
check("root honours $EBSW_ARTIFACT_ROOT", lu.write_root() == os.environ["EBSW_ARTIFACT_ROOT"])

# --- directories are created on demand ------------------------------------------------
p = lu.out("a/b/c/file.json")
check("nested output directories created", os.path.isdir(os.path.dirname(p)))

# --- JSON round-trip with awkward types -----------------------------------------------
payload = {
    "floats": [1.5, 2.5],
    "tensor_like": torch.tensor([1.0, 2.0]),
    "nested": {"x": None, "y": True},
    "string": "ok",
}
lu.save_json(payload, "a/b/c/file.json")
back = lu.load_json("a/b/c/file.json")
check("json round-trip keeps nested structure", back["nested"] == {"x": None, "y": True})
check("json round-trip handles objects with tolist()", back["tensor_like"] == [1.0, 2.0])

# --- JSONL -----------------------------------------------------------------------------
rows = [{"prompt_id": f"p{i}", "token_ids": list(range(i))} for i in range(5)]
lu.write_jsonl(rows, "calib/test.jsonl")
back_rows = lu.read_jsonl("calib/test.jsonl")
check("jsonl round-trip", back_rows == rows, f"{len(back_rows)} rows")
lu.append_jsonl({"stage": "test"}, "logs/run_log.jsonl")
lu.append_jsonl({"stage": "test2"}, "logs/run_log.jsonl")
check("jsonl append", len(lu.read_jsonl("logs/run_log.jsonl")) == 2)

# --- require() must name the producing notebook ----------------------------------------
try:
    lu.require("calib/missing.jsonl", produced_by="99_never_run.ipynb")
    check("require raises on a missing artifact", False)
except FileNotFoundError as exc:
    msg = str(exc)
    check("require raises FileNotFoundError", True)
    check("error names the producing notebook", "99_never_run.ipynb" in msg)
    check("error lists the search roots", "searched" in msg.lower())

check("find returns None for a missing artifact", lu.find("calib/missing.jsonl") is None)
check("find locates an existing artifact", lu.find("calib/test.jsonl") is not None)

# --- artifact discovery from an attached read-only root --------------------------------
attached = os.path.join(TMP, "input", "prev_notebook", "artifacts")
os.makedirs(os.path.join(attached, "calib"), exist_ok=True)
with open(os.path.join(attached, "calib", "heldout.jsonl"), "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"prompt_id": "h"}) + "\n")

real_glob = lu.glob.glob
lu.glob.glob = lambda pattern: ([str(attached)] if pattern == "/kaggle/input/*/artifacts"
                                else real_glob(pattern))
found = lu.find("calib/heldout.jsonl")
lu.glob.glob = real_glob
check("find falls back to an attached notebook output", found is not None, str(found))

# --- zip bundle ------------------------------------------------------------------------
zip_path = lu.zip_artifacts("test_bundle.zip", ["calib/test.jsonl"])
check("zip created", os.path.exists(zip_path), zip_path)
import zipfile

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
check("zip keeps artifacts/ prefix (so it re-attaches cleanly)",
      any(n.startswith("artifacts/calib/test.jsonl") for n in names), str(names))

# --- environment manifest --------------------------------------------------------------
man = lu.environment_manifest({"extra": 1})
check("manifest has platform and torch blocks",
      "platform" in man and "torch" in man and man["extra"] == 1)
check("manifest records package versions", isinstance(man["packages"], dict))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL ARTIFACT TESTS PASSED")
print("temp dir:", TMP)
