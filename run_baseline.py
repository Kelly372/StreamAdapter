"""Launch unmodified LongLive sampling with portable config and run records."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf
from utils.project_paths import REPO_ROOT, add_path_arguments, load_config, resolve_path
from utils.run_record import environment_record, write_json
from utils.next_command import next_run_name, print_next_command


def validate_baseline(config):
    if config.get("experiment", {}).get("status") == "spec_only":
        raise ValueError("Camera training is not implemented. Select configs/baseline/*.yaml.")
    if config.get("adapter") or config.get("camera_adapter", {}).get("enabled", False):
        raise ValueError("Baseline reproduction requires a merged generator and no additional Adapter.")
    if not config.get("generator_ckpt"):
        raise ValueError("checkpoints.generator_ckpt is required; base Wan weights alone are not LongLive.")
    frames = int(config.num_output_frames)
    if frames != int(config.image_or_video_shape[1]) or frames % int(config.num_frame_per_block):
        raise ValueError("num_output_frames must equal data.image_or_video_shape[1] and be block-aligned.")
    if config.get("i2v") and not config.get("independent_first_frame"):
        raise ValueError("I2V baseline requires inference.independent_first_frame=true.")
    if config.get("fp8_quant") and config.get("model_quant"):
        raise ValueError("FP8 and NVFP4 cannot be enabled together.")


def check_inputs(config):
    errors = []
    for key, is_dir in (("base_model_dir", True), ("tokenizer_path", True),
                        ("text_encoder_path", False), ("vae_path", False),
                        ("generator_ckpt", False)):
        value = config.get(key)
        path = Path(value) if value else None
        if path is None or not (path.is_dir() if is_dir else path.is_file()):
            errors.append(f"{key}: {value or '<unset>'}")
    base = Path(config.base_model_dir)
    if base.is_dir():
        if not (base / "config.json").is_file():
            errors.append(f"base model config: {base / 'config.json'}")
        if not any(base.glob("*.safetensors")) and not any(base.glob("*.bin")):
            errors.append(f"base model weights (*.safetensors or *.bin): {base}")
    data = Path(config.data_path)
    if not data.exists():
        errors.append(f"data_path: {data}")
    elif config.get("i2v") and not data.is_dir():
        errors.append("I2V data_path must be an image/prompt directory. Use prepare_realcam_windows.py to export baseline_i2v/ from an inspected CSV manifest.")
    elif not config.get("i2v") and data.is_file():
        if data.suffix.lower() != ".txt" or not data.read_text(encoding="utf-8").strip():
            errors.append("T2V data_path must be a nonempty prompt .txt or a caption directory.")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline/longlive_bf16_i2v.yaml")
    add_path_arguments(parser)
    parser.add_argument("--run-name", help="Unique output subdirectory name; never overwrites an existing run")
    parser.add_argument("--tag", default="", help="Optional experiment label appended to the generated name")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and record config, report inputs; no GPU/model imports")
    parser.add_argument("--check-only", action="store_true", help="Check required paths; return nonzero if any are missing")
    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        raise ValueError("run_baseline.py is a single-process launcher; do not launch it with torchrun.")
    config = load_config(args.config, workspace=args.workspace_root, overrides=args.overrides)
    validate_baseline(config)
    mode = "i2v" if config.get("i2v", False) else "t2v"
    precision = "nvfp4" if config.get("model_quant") else "fp8" if config.get("fp8_quant") else "bf16"
    automatic_name = (f"baseline_{mode}_longlive2_{precision}_s{config.sampling_steps}_"
                      f"f{config.num_output_frames}_seed{config.seed}_"
                      f"{datetime.now():%Y%m%d-%H%M%S-%f}")
    run_name = args.run_name or automatic_name + (f"_{args.tag}" if args.tag else "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_name) or run_name.endswith("."):
        raise ValueError("Run name must be a portable single folder name: letters, digits, _, -, . (max 180).")
    if run_name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("Run name is reserved on Windows.")
    folder = REPO_ROOT / "output" / run_name
    folder.mkdir(parents=True, exist_ok=False)
    config.output_folder = str(folder)
    config.inference.output_folder = str(folder)
    config.run = {"name": run_name, "record_parameters": True}
    config_path = folder / "resolved_config.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    source = resolve_path(args.config, REPO_ROOT)
    (folder / "source_config.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    command = [sys.executable, "-u", str(REPO_ROOT / "inference.py"), "--config_path", str(config_path)]
    # argv arrays, rather than a shell string, preserve spaces on both platforms.
    record = {"launcher_argv": sys.argv, "child_argv": command, "cwd": str(REPO_ROOT),
              "overrides": args.overrides, "environment": environment_record()}
    write_json(folder / "launch.json", record)
    (folder / "inference.txt").write_text(
        "LongLive 2.0 baseline reproduction\n\n" +
        "Resolved parameters (runtime defaults appended after model initialization):\n" +
        OmegaConf.to_yaml(config, resolve=True) + "\nLaunch/environment: see launch.json\n", encoding="utf-8")
    missing = check_inputs(config)
    write_json(folder / "input_check.json", {"ok": not missing, "missing_or_invalid": missing})
    print(f"Run directory: {folder}", flush=True)
    for error in missing:
        print(f"Missing/invalid input: {error}", flush=True)
    if args.dry_run or args.check_only:
        status = "dry_run" if args.dry_run else "inputs_valid" if not missing else "inputs_missing"
        write_json(folder / "status.json", {"status": status, "gpu_tested": False})
        if not missing:
            command = ["python", "run_baseline.py", "--config", args.config]
            if args.workspace_root is not None:
                command += ["--workspace-root", args.workspace_root]
            for override in args.overrides:
                command += ["--set", override]
            if args.tag:
                command += ["--tag", args.tag]
            if args.dry_run:
                command += ["--check-only"]
            command += ["--run-name", next_run_name(folder, "check", "check" if args.dry_run else "baseline")]
            print_next_command(folder, command, REPO_ROOT)
        return 1 if args.check_only and missing else 0
    if missing:
        write_json(folder / "status.json", {"status": "inputs_missing", "gpu_tested": False})
        return 1
    write_json(folder / "status.json", {"status": "running"})
    try:
        # Inherit ordinary runtime environment; launcher requires no manual export.
        with (folder / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    except BaseException as exc:
        write_json(folder / "status.json", {"status": "interrupted_or_failed", "error": str(exc)})
        raise
    write_json(folder / "status.json", {"status": "completed" if returncode == 0 else "failed", "returncode": returncode})
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
