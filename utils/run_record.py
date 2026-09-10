"""Experiment records used by the baseline launcher and inference process."""
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from omegaconf import OmegaConf
from utils.project_paths import REPO_ROOT


def environment_record():
    versions = {}
    for package in ("torch", "torchvision", "omegaconf", "diffusers", "transformers",
                    "flash-attn", "torchao", "transformer-engine", "fouroversix"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "runtime_environment": {key: value for key, value in os.environ.items()
                                    if key.startswith("LLV2_") or key in
                                    ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")}}


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def _plain(value):
    """Normalize library config values such as tuples, sets and dtype objects."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def save_runtime_record(config, pipeline, device, num_prompts):
    """Record runtime defaults after pipeline creation, before generation."""
    import torch

    folder = Path(config.output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    scalar_types = (str, int, float, bool, type(None))
    resolved_pipeline = {key: value for key, value in vars(pipeline).items()
                         if not key.startswith("_") and isinstance(value, scalar_types)}
    gpu = torch.cuda.get_device_properties(device)
    scheduler = pipeline._initialize_sample_scheduler(torch.empty(0, device=device))
    record = {"config": OmegaConf.to_container(config, resolve=True),
              "pipeline_parameters": resolved_pipeline,
              "model_configuration": dict(pipeline._dit_model.config),
              "scheduler": dict(scheduler.config),
              "sampling_timesteps": scheduler.timesteps.detach().cpu().tolist(),
              "sampling_sigmas": scheduler.sigmas.detach().cpu().tolist(),
              "num_prompts": num_prompts, "device": str(device), "gpu_name": gpu.name,
              "gpu_memory_bytes": gpu.total_memory,
              "cuda_version": torch.version.cuda, "environment": environment_record()}
    # Scheduler configurations contain simple scalar/list settings.
    record = _plain(record)
    OmegaConf.save(OmegaConf.create(record), folder / "runtime_config.yaml")
    with (folder / "inference.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n\nRuntime parameters (including pipeline/scheduler defaults):\n")
        handle.write(OmegaConf.to_yaml(OmegaConf.create(record), resolve=True))
