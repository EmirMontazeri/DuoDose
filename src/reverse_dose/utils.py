"""Atomic artifacts, deterministic execution, and persistent experiment logging."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_default(value: Any) -> Any:
    """Convert common scientific scalar/path types without allowing NaN JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for experiment identities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    )


def fingerprint(value: Any) -> str:
    """SHA256 of canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact after flushing its complete content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    """Write a checkpoint atomically; leave the previous checkpoint intact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a locally generated checkpoint including Python/NumPy RNG state.

    PyTorch checkpoints are pickle containers; this API is for trusted project
    artifacts, never downloaded untrusted pickle files.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def setup_logging(directory: Path | None = None) -> None:
    """Log stage and batch progress to both the terminal and a persistent file."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(directory / "pipeline.log", encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)


def event(path: Path, payload: dict[str, Any]) -> None:
    """Append a durable JSONL event with an explicit UTC timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            canonical_json({"time": datetime.now(timezone.utc).isoformat(), **payload}) + "\n"
        )
        handle.flush()


def seed_everything(seed: int, deterministic: bool = True, threads: int = 4) -> None:
    """Seed all model RNGs. CPU execution is the strict reference backend."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def resolve_device(requested: str) -> torch.device:
    """Choose an explicitly available CPU/CUDA/MPS device."""
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(requested)


def rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU, CUDA, and MPS random number generators."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Restore a checkpoint's RNG state after model/optimizer construction."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def environment_info() -> dict[str, Any]:
    """Capture runtime versions and source code hashes for reproducibility."""
    packages = {}
    for name in (
        "reverse-dose",
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "rdkit",
        "scipy",
        "openpyxl",
        "PyYAML",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    code = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_hashes": code,
        "cuda": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
    }


def guard_output(path: Path, dataset_root: Path) -> None:
    """Refuse artifact paths inside immutable raw data (including symlink targets)."""
    raw = (dataset_root / "raw").resolve()
    destination = path.resolve()
    if destination == raw or raw in destination.parents:
        raise ValueError(f"Output path is inside immutable raw data: {path}")
