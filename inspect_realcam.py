"""Inspect RealCam-Vid CSVs and build portable I2V metadata manifests (CPU)."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

from omegaconf import OmegaConf

from utils.project_paths import REPO_ROOT, add_path_arguments, resolve_path, workspace_root
from utils.realcam_inspection import inspect_dataset
from utils.run_record import environment_record, write_json
from utils.next_command import data_context, next_run_name, print_next_command


def load_inspection_config(path, workspace=None, overrides=()):
    # Use shared path primitives without the model-specific config normalizer.
    config = OmegaConf.merge(OmegaConf.load(resolve_path(path, REPO_ROOT)),
                             OmegaConf.from_dotlist(list(overrides)))
    root = workspace_root(workspace if workspace is not None else config.paths.workspace_root)
    config.paths.workspace_root = str(root)
    config.paths.repo_root = str(REPO_ROOT)
    config.paths.data_root = str(resolve_path(config.paths.data_root, root))
    for split in ("train", "test"):
        for suffix in ("csv", "camera_npz"):
            key = f"{split}_{suffix}"
            if config.data.get(key):
                config.data[key] = str(resolve_path(config.data[key], config.paths.data_root))
    OmegaConf.resolve(config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/realcam_inspection.yaml")
    add_path_arguments(parser)
    parser.add_argument("--probe-mode", choices=("decode", "header", "metadata"))
    parser.add_argument("--camera-metadata-dir", help="Extracted NPZ directory, relative to repository; selected files must exist")
    parser.add_argument("--limit", type=int, help="Camera/video checks per split; 0 = all; source/directory groups scanned globally")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-name", help="Unique output subfolder; never overwritten")
    args = parser.parse_args(argv)
    overrides = list(args.overrides)
    if args.probe_mode is not None:
        overrides.append(f"inspection.probe_mode={args.probe_mode}")
    if args.limit is not None:
        overrides.append(f"inspection.limit_per_split={args.limit}")
    config = load_inspection_config(args.config, args.workspace_root, overrides)
    if args.camera_metadata_dir:
        directory = resolve_path(args.camera_metadata_dir, REPO_ROOT)
        for split in ("train", "test"):
            config.data[f"{split}_camera_npz"] = str(directory / f"{config.data.subset}_{split}.npz")
    name = args.run_name or (f"data_check_realestate10k_{config.inspection.probe_mode}_"
                             f"seed{config.seed}_{datetime.now():%Y%m%d-%H%M%S-%f}" +
                             (f"_{args.tag}" if args.tag else ""))
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", name) or name.endswith(".")
            or name.split(".")[0].upper() in reserved):
        raise ValueError("Run name must be a portable single folder name (letters, digits, _, -, .; max 180)")
    folder = REPO_ROOT / "output" / name
    folder.mkdir(parents=True, exist_ok=False)
    config.run = {"name": name, "output_folder": str(folder)}
    OmegaConf.save(config, folder / "resolved_config.yaml", resolve=True)
    (folder / "source_config.yaml").write_text(resolve_path(args.config, REPO_ROOT).read_text(encoding="utf-8"), encoding="utf-8")
    write_json(folder / "launch.json", {"argv": list(sys.argv if argv is None else argv),
                                      "effective_overrides": overrides, "environment": environment_record()})
    (folder / "inspection.txt").write_text("RealCam-Vid CSV inspection (no window sampling or camera transformation)\n\n" +
                                         OmegaConf.to_yaml(config, resolve=True) +
                                         "\nEnvironment: launch.json; discovered files/columns/hashes: metadata_sources.json\n",
                                         encoding="utf-8")
    print(f"Run directory: {folder}", flush=True)
    write_json(folder / "status.json", {"status": "running"})
    try:
        summary = inspect_dataset(config, folder)
        failures = []
        if not summary["accepted"]["train"] or not summary["accepted"]["test"]:
            failures.append("No accepted training or test samples; inspect rejections and validation_fraction")
        if summary["overlapping_train_test_groups"] and config.inspection.fail_on_leakage:
            failures.append("Source or directory groups overlap between train and test; inspect grouping_bases_in_scanned_rows")
        if summary["rejected"] and config.inspection.fail_on_reject:
            failures.append("Rows were rejected and inspection.fail_on_reject=true")
        code = 1 if failures else 0
        write_json(folder / "status.json", {"status": "audit_failed" if failures else "completed_with_rejections" if summary["rejected"] else "completed",
                                          "returncode": code, "failures": failures,
                                          "probe_mode": config.inspection.probe_mode,
                                          "original_source_leakage_check_complete": summary["original_source_leakage_check_complete"],
                                          "all_selected_rows_inspected": summary["all_selected_rows_inspected"]})
        with (folder / "inspection.txt").open("a", encoding="utf-8") as handle:
            handle.write("\nResults:\n" + (folder / "summary.json").read_text(encoding="utf-8"))
        print(f"Accepted: {summary['accepted']}; rejected: {summary['rejected']}; mode: {summary['probe_mode']}", flush=True)
        if not summary["original_source_leakage_check_complete"]:
            print("Grouping includes directories or unidentified rows; original-video-level leakage check is incomplete. See summary.json.", flush=True)
        for failure in failures:
            print(f"Audit failed: {failure}", flush=True)
        if code == 0:
            command = ["python", "prepare_realcam_windows.py", "--manifest", f"output/{name}/train.csv",
                       "--limit", "10", "--export-count", "2", "--run-name", next_run_name(folder, "audit", "windows")]
            command += data_context(config, REPO_ROOT)
            for key, value, default in (("seed", config.seed, 0), ("windows.rgb_frames", config.data.rgb_frames, 125),
                                        ("windows.target_fps", config.data.target_fps, 24)):
                if value != default:
                    command += ["--set", f"{key}={value}"]
            print_next_command(folder, command, REPO_ROOT)
        return code
    except (Exception, KeyboardInterrupt) as exc:
        write_json(folder / "status.json", {"status": "failed", "returncode": 2,
                                          "error_type": type(exc).__name__, "error": str(exc)})
        print(f"Inspection failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
