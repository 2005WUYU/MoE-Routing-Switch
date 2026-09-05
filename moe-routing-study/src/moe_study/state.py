"""One-step snapshots and a small CPU development checkpoint implementation.

The cluster adapter uses Bridge's native distributed checkpoint, not these
single-process files. A checkpoint is written only at a completed update.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import shutil
import subprocess
import zipfile

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class Progress:
    step: int = 0
    consumed_sequences: int = 0


def snapshot_parameters(model: nn.Module) -> dict[str, Tensor]:
    """CPU development model parameters are already FP32 master parameters."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def interpolate_parameters(old: dict[str, Tensor], new: dict[str, Tensor], alpha: float) -> dict[str, Tensor]:
    return {name: value + alpha * (new[name] - value) for name, value in old.items()}


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class CPUCheckpoint:
    def __init__(self, root: Path):
        self.root = root

    def save(self, model: nn.Module, optimizer, progress: Progress) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        pointer = self.root / "latest.json"
        previous = json.loads(pointer.read_text())["directory"] if pointer.exists() else None
        directory = f"step_{progress.step:06d}"
        destination = self.root / directory
        destination.mkdir(exist_ok=True)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "progress": asdict(progress), "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        }, destination / "state.pt")
        pointer.write_text(json.dumps({"directory": directory, **asdict(progress)}, indent=2))
        if previous is not None and previous != directory:
            shutil.rmtree(self.root / previous)
        return (destination / "state.pt").stat().st_size

    def load(self, model: nn.Module, optimizer) -> Progress:
        directory = json.loads((self.root / "latest.json").read_text())["directory"]
        state = torch.load(self.root / directory / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        return Progress(**state["progress"])


def record_code(directory: Path) -> dict:
    """Record dirty work normally. Git metadata is optional for an unpacked directory."""
    root = Path(__file__).resolve().parents[2]
    snapshot = directory / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "tests", "experiments", "machines", "schedules", "launch", "docs", "examples"):
            for path in (root / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(root))
        for name in ("pyproject.toml", "README.md"):
            archive.write(root / name, name)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if commit.returncode == 0:
        patch = subprocess.run(["git", "diff", "HEAD", "--binary"], capture_output=True, text=True, check=True)
        (directory / "working-tree.patch").write_text(patch.stdout)
        return {"commit": commit.stdout.strip(), "patch": "working-tree.patch", "source_snapshot": snapshot.name}
    return {"commit": None, "source": "unversioned_directory", "source_snapshot": snapshot.name}
