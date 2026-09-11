"""Validate a prepared RealEstate10K dataset and compare I2V results in one run directory."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import inspect_realcam
import prepare_realcam_windows
import run_baseline
from utils.baseline_reference import copy_reference, digest
from utils.next_command import portable_path, print_next_command
from utils.project_paths import REPO_ROOT, resolve_path, workspace_root, stage_output_dir, load_config
from utils.run_record import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_snapshot(data):
    return {p.name: digest(p) for p in sorted(Path(data).iterdir()) if p.is_file()}


def show_summary(folder, summary):
    write_json(folder / "summary.json", summary)
    (folder / "validation.txt").write_text(
        "RealEstate10K I2V validation\n"
        "Open comparisons/0000/, 0001/, ... to compare ground_truth.mp4 and generated.mp4.\n"
        "input.png is the conditioning frame; sample.json records the selected window.\n"
        "GT is a cropped/resampled preview. Camera trajectories are not model inputs yet.\n"
        "Parameters, effective configs, manifests, rejections and logs: records/\n\n" +
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validation directory: {folder}\nStatus: {summary['status']}", flush=True)


def infer(folder):
    summary = read_json(folder / "summary.json")
    if summary["status"] != "prepared":
        raise ValueError("Only a prepared run can start inference; use a new run name for a new validation")
    records = folder / "records"
    parameters = read_json(records / "parameters.json")
    data = records / "windows" / "baseline_i2v"
    if input_snapshot(data) != read_json(records / "input_snapshot.json"):
        raise ValueError("Prepared input images/prompts changed; start a new validation")
    summary["status"] = "inference_running"
    show_summary(folder, summary)
    # There is no separate check-only run. The normal launcher checks inputs before loading the model.
    args = ["--config", portable_path(records / "baseline_config.yaml", REPO_ROOT),
            "--workspace-root", parameters["workspace_root"],
            "--set", f"data.data_path={portable_path(data, REPO_ROOT)}",
            "--set", f"logging.seed={parameters['seed']}", "--set", "num_samples=1",
            "--set", "inference_iter=-1", "--set", "inference.save_latents_only=false",
            "--set", f"inference.comparison_dir={portable_path(folder / 'comparisons', REPO_ROOT)}"]
    try:
        code = run_baseline.main(args, output_dir=records / "inference", show_next=False)
        outputs = sorted((folder / "comparisons").glob("*/generated.mp4"))
        summary.update(generated=len(outputs), status="completed" if code == 0 and len(outputs) == summary["exported"] else "inference_failed")
        show_summary(folder, summary)
        return 0 if summary["status"] == "completed" else (code or 1)
    except BaseException as exc:
        summary.update(status="inference_failed", error=str(exc))
        show_summary(folder, summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", help="Unique folder under output/, for example realestate10k_v1")
    parser.add_argument("--infer", metavar="RUN_DIR", help="Generate videos for this prepared run; relative to repository")
    parser.add_argument("--workspace-root")
    parser.add_argument("--data-root", default="public_data/RealCam-Vid/RealEstate10K")
    parser.add_argument("--baseline-config", default="configs/baseline/longlive_bf16_i2v.yaml")
    parser.add_argument("--count", type=int, default=10, help="Number of distinct clips to export and generate")
    parser.add_argument("--limit", type=int, default=50, help="Inspection candidates per split, 0=all")
    parser.add_argument("--split", choices=("train", "test"), default="test", help="Baseline comparison split; default test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.infer:
        try:
            return infer(stage_output_dir(args.infer, REPO_ROOT))
        except (Exception, KeyboardInterrupt) as exc:
            print(f"Inference could not complete: {exc}", file=sys.stderr, flush=True)
            return 1
    if args.count < 1 or args.limit < 0 or (args.limit and args.limit < args.count):
        parser.error("--count must be positive; --limit must be 0 or at least --count")
    name = args.run_name or f"realestate10k_n{args.count}_seed{args.seed}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name)
            or name.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}):
        parser.error("Use a portable run name of up to 100 letters, digits, underscores or hyphens")
    folder = REPO_ROOT / "output" / name
    folder.mkdir(parents=True, exist_ok=False)
    records = folder / "records"
    records.mkdir()
    summary = {"status": "preparing", "requested": args.count, "split": args.split, "exported": 0,
               "generated": 0, "camera_conditioning": False, "full_dataset_validation": args.limit == 0}
    show_summary(folder, summary)
    root = workspace_root(args.workspace_root)
    data_root = resolve_path(args.data_root, root)
    parameters = dict(vars(args), workspace_root=portable_path(root, REPO_ROOT),
                      resolved_data_root=str(data_root), argv=argv if argv is not None else sys.argv[1:])
    write_json(records / "parameters.json", parameters)
    try:
        # Dataset construction is complete: require exactly these four prepared files, no raw-archive fallback.
        for split in ("train", "test"):
            for suffix in ("csv", "npz"):
                path = data_root / f"RealEstate10K_{split}.{suffix}"
                if not path.is_file():
                    raise ValueError(f"Prepared dataset file missing: {path}")
        baseline_path = resolve_path(args.baseline_config, REPO_ROOT)
        (records / "baseline_config.yaml").write_text(baseline_path.read_text(encoding="utf-8"), encoding="utf-8")
        baseline = load_config(str(baseline_path), workspace=str(root))
        run_baseline.validate_baseline(baseline)
        if not baseline.get("i2v") or baseline.model_kwargs.model_name != "Wan2.2-TI2V-5B":
            raise ValueError("This validation entry requires a Wan2.2-TI2V-5B I2V baseline config")
        shape = baseline.image_or_video_shape
        rgb_frames = (int(baseline.num_output_frames) - 1) * 4 + 1
        common = ["--workspace-root", str(root), "--set", f"paths.data_root={data_root.as_posix()}",
                  "--set", f"seed={args.seed}"]
        inspection_args = common + ["--probe-mode", "decode", "--limit", str(args.limit),
                                    "--set", f"data.rgb_frames={rgb_frames}"]
        for split in ("train", "test"):
            inspection_args += ["--set", f"data.{split}_csv=RealEstate10K_{split}.csv",
                                "--set", f"data.{split}_camera_npz=RealEstate10K_{split}.npz"]
        inspection = records / "inspection"
        code = inspect_realcam.main(inspection_args, output_dir=inspection, show_next=False)
        summary["inspection"] = read_json(inspection / "summary.json") if (inspection / "summary.json").is_file() else {}
        if code:
            raise ValueError("Data inspection failed; see records/inspection/status.json and rejected.csv")
        windows = records / "windows"
        window_args = common + ["--manifest", str(inspection / f"{args.split}.csv"),
                                "--limit", "0", "--export-count", str(args.count),
                                "--set", f"windows.rgb_frames={rgb_frames}",
                                "--set", f"windows.height={int(shape[-2]) * 16}",
                                "--set", f"windows.width={int(shape[-1]) * 16}"]
        code = prepare_realcam_windows.main(window_args, output_dir=windows, show_next=False)
        window_summary = read_json(windows / "summary.json") if (windows / "summary.json").is_file() else {}
        summary["windows"] = window_summary
        summary["exported"] = window_summary.get("exported_samples", 0)
        if code:
            raise ValueError("Window preparation failed; see records/windows/status.json and rejected.json")
        images = sorted((windows / "baseline_i2v").glob("*.png"), key=lambda p: p.name)
        for index, image in enumerate(images):
            comparison = folder / "comparisons" / f"{index:04d}"
            comparison.mkdir(parents=True)
            reference = copy_reference(image, comparison, "generated", simple_names=True)
            if reference["status"] != "gt_copied":
                raise ValueError(f"No GT for prepared image: {image}")
        write_json(records / "input_snapshot.json", input_snapshot(windows / "baseline_i2v"))
        if summary["exported"] != args.count:
            raise ValueError(f"Requested {args.count} clips, found {summary['exported']}. Use a new run name with a larger --limit")
        summary["status"] = "prepared"
        show_summary(folder, summary)
        print_next_command(folder, ["python", "validate_realestate10k.py", "--infer", portable_path(folder, REPO_ROOT)], REPO_ROOT)
        (folder / "next_command.json").replace(records / "next_command.json")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="preparation_failed", error=str(exc))
        show_summary(folder, summary)
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
