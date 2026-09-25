#!/usr/bin/env python3
"""Dump GENCO benchmark venv + node versions (no timed work)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PKGS = [
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "torch-geometric",
    "torch-scatter",
    "gridfm-graphkit",
    "lightning",
    "numpy",
    "pandas",
    "scipy",
    "PyYAML",
    "tqdm",
]

NVIDIA_PKGS = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-nccl-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-nvjitlink-cu12",
]


def pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"(failed: {exc})"


def main() -> None:
    import torch

    info: dict = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "lsf_job": os.environ.get("LSB_JOBID"),
        "lsf_queue": os.environ.get("LSB_QUEUE"),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "packages": {name: pkg_version(name) for name in PKGS},
        "nvidia_wheels": {name: pkg_version(name) for name in NVIDIA_PKGS},
        "nvidia_smi": run(["nvidia-smi"]),
        "lscpu_model": run(["bash", "-lc", "lscpu | grep -E 'Model name|CPU\\(s\\)|Socket'"]),
        "uname": run(["uname", "-a"]),
    }

    out_dir = Path(os.environ.get("GENCO_ENV_DUMP_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "genco_environment_dump.json"
    json_path.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
