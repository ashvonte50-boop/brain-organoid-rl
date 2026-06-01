"""
extensions/repro.py — Reproducibility infrastructure (Task 6).

Provides:
  snapshot_config     — capture all cf.* module constants into a dict
  save_manifest       — write JSON manifest: config + git hash + file checksums
  load_manifest       — load and validate a saved manifest
  hash_array          — deterministic sha256 of a numpy array
  hash_results        — hash a nested results dict (arrays + scalars)
  validate_repro      — compare two result hashes, report differences
  ResultManifest      — dataclass bundling config + results hash + metadata

All outputs are plain JSON / numpy-safe dicts — no pickle.
"""
import hashlib
import json
import os
import struct
import subprocess
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "snapshot_config",
    "save_manifest",
    "load_manifest",
    "hash_array",
    "hash_results",
    "validate_repro",
    "ResultManifest",
]


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

_CONFIG_KEYS = [
    "N_NEURONS", "N_EXC", "N_INH", "DT",
    "MASTER_SEED", "GAMMA", "FAST_DECAY_TAU",
    "N_MEMORIES", "ASSEMBLY_SIZE", "CUE_SIZE", "PARTIAL_CUE_SIZE",
    "STIM_STRENGTH", "CUE_STRENGTH",
    "N_TRIALS", "N_TRIALS_SWEEP", "N_TRIALS_ABLATION",
    "REPLAY_NOISE_STD", "REPLAY_BURST_SIZE", "REPLAY_BURST_GAP",
    "REPLAY_COHERENCE_THR", "REPLAY_COHERENCE_LAMBDA",
    "TAG_CAPTURE_RATE", "W_MAX",
    "INTER_MEM_REST_STEPS",
    "REPLAY_PERS_GAIN", "REPLAY_PERS_DECAY", "REPLAY_PERS_BUDGET",
    "DEV_MODE",
]


def snapshot_config(cf_module=None) -> Dict[str, Any]:
    """
    Return a JSON-serialisable dict of all tracked cf.* constants.
    Pass the compare_catastrophic_forgetting module, or None to import it.
    """
    if cf_module is None:
        import compare_catastrophic_forgetting as cf_module
    cfg = {}
    for key in _CONFIG_KEYS:
        val = getattr(cf_module, key, None)
        if val is None:
            continue
        if isinstance(val, (bool, int, float, str)):
            cfg[key] = val
        elif isinstance(val, np.integer):
            cfg[key] = int(val)
        elif isinstance(val, np.floating):
            cfg[key] = float(val)
        else:
            cfg[key] = str(val)
    # dynamic derived constants
    try:
        cfg["_N_PRESENTATIONS"] = int(cf_module._N_PRESENTATIONS)
        cfg["_N_REPLAY_EVENTS"] = int(cf_module._N_REPLAY_EVENTS)
    except AttributeError:
        pass
    return cfg


# ---------------------------------------------------------------------------
# File checksums
# ---------------------------------------------------------------------------

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Array / result hashing
# ---------------------------------------------------------------------------

def hash_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array in C-contiguous float64 layout."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def hash_results(results: Any, _h: Optional[hashlib._hashlib.HASH] = None) -> str:
    """
    Deterministic hash of a nested structure of dicts, lists, numpy arrays,
    and scalars. Returns 16-char hex prefix.
    """
    if _h is None:
        _h = hashlib.sha256()
        hash_results(results, _h)
        return _h.hexdigest()[:16]

    if isinstance(results, dict):
        for k in sorted(results.keys()):
            _h.update(str(k).encode())
            hash_results(results[k], _h)
    elif isinstance(results, (list, tuple)):
        for item in results:
            hash_results(item, _h)
    elif isinstance(results, np.ndarray):
        _h.update(np.ascontiguousarray(results, dtype=np.float64).tobytes())
    elif isinstance(results, (int, float, bool)):
        _h.update(struct_pack(results))
    elif isinstance(results, str):
        _h.update(results.encode())
    elif results is None:
        _h.update(b"\x00")
    else:
        _h.update(str(results).encode())
    return ""


def struct_pack(val) -> bytes:
    if isinstance(val, bool):
        return struct.pack("?", val)
    if isinstance(val, int):
        return struct.pack("<q", val)
    if isinstance(val, float):
        return struct.pack("<d", val)
    return str(val).encode()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ResultManifest:
    run_id: str
    timestamp: str
    git_hash: str
    git_dirty: bool
    config: Dict[str, Any]
    results_hash: str
    file_checksums: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResultManifest":
        return cls(**d)


def save_manifest(
    results: Any,
    out_path: str,
    cf_module=None,
    extra_files: Optional[List[str]] = None,
    notes: str = "",
    run_id: Optional[str] = None,
) -> ResultManifest:
    """
    Build and save a ResultManifest JSON to out_path.

    Args:
        results      : the experiment results structure to hash
        out_path     : where to write the .json manifest
        cf_module    : compare_catastrophic_forgetting module (or None to import)
        extra_files  : additional file paths to checksum
        notes        : free-text annotation
        run_id       : override run identifier (default: timestamp-based)
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    rid = run_id or f"run_{ts.replace(':', '').replace('-', '').replace('T', '_')}"

    cfg = snapshot_config(cf_module)
    rhash = hash_results(results)
    git_hash = _git_head()
    dirty = _git_dirty()

    # Checksum core script + any extra files
    checksums: Dict[str, str] = {}
    targets = ["compare_catastrophic_forgetting.py"] + (extra_files or [])
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for rel in targets:
        p = os.path.join(root, rel) if not os.path.isabs(rel) else rel
        if os.path.isfile(p):
            checksums[rel] = _file_sha256(p)

    manifest = ResultManifest(
        run_id=rid,
        timestamp=ts,
        git_hash=git_hash,
        git_dirty=dirty,
        config=cfg,
        results_hash=rhash,
        file_checksums=checksums,
        notes=notes,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return manifest


def load_manifest(path: str) -> ResultManifest:
    """Load a manifest JSON and return a ResultManifest object."""
    with open(path) as f:
        d = json.load(f)
    return ResultManifest.from_dict(d)


def validate_repro(
    results: Any,
    manifest_path: str,
    warn: bool = True,
) -> Tuple:
    """
    Re-hash results and compare against a saved manifest.

    Returns (match: bool, new_hash: str, expected_hash: str).
    """
    manifest = load_manifest(manifest_path)
    new_hash = hash_results(results)
    match = new_hash == manifest.results_hash
    if not match and warn:
        warnings.warn(
            f"Reproducibility check FAILED:\n"
            f"  expected: {manifest.results_hash}\n"
            f"  got:      {new_hash}\n"
            f"  manifest: {manifest_path}"
        )
    return match, new_hash, manifest.results_hash


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os, sys as _sys, pathlib as _pl
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Ensure project root (parent of extensions/) is on sys.path
    _root = str(_pl.Path(__file__).resolve().parent.parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    rng = np.random.default_rng(0)
    fake_results = {
        "condition_A": {"scores": rng.normal(0.8, 0.1, 15)},
        "condition_B": {"scores": rng.normal(0.1, 0.1, 15)},
    }

    manifest = save_manifest(
        fake_results,
        out_path="/tmp/test_manifest.json",
        notes="self-test run",
        run_id="selftest_001",
    )
    print(f"Saved manifest: run_id={manifest.run_id}, results_hash={manifest.results_hash}")
    print(f"  git_hash={manifest.git_hash}, dirty={manifest.git_dirty}")
    print(f"  config keys: {list(manifest.config.keys())[:5]}...")

    match, new_h, exp_h = validate_repro(fake_results, "/tmp/test_manifest.json", warn=False)
    print(f"Repro check: match={match}, hashes: {new_h} == {exp_h}")

    # Mutate and check mismatch
    fake_results["condition_A"]["scores"][0] += 1.0
    match2, _, _ = validate_repro(fake_results, "/tmp/test_manifest.json", warn=False)
    print(f"After mutation: match={match2}  (expected False)")
