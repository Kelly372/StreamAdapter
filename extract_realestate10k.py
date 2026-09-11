"""Step 3A: build CSV-aligned RealEstate10K NPZs, or inspect original archives."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np

from utils.project_paths import REPO_ROOT, resolve_path, workspace_root
from utils.realcam_dataset import discover_csv, load_official_camera_index, sha256_file
from utils.run_record import environment_record, write_json
from utils.realcam_readable import export_readable_npz
from utils.realcam_partition import partition_by_csv
from utils.next_command import portable_path, print_next_command


def metadata_equal(left, right):
    """Check all fields, including array dtype/shape/content, after saving."""
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape
                and (all(metadata_equal(a, b) for a, b in zip(left.flat, right.flat)) if left.dtype.hasobject
                     else left.tobytes() == right.tobytes()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(metadata_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(metadata_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)) and np.isnan(left):
        return isinstance(right, (float, np.floating)) and np.isnan(right)
    return type(left) is type(right) and bool(left == right)


def extract_split(source, destination, data_root):
    original_hash = sha256_file(source)
    index = load_official_camera_index(source, data_root)
    selected = []
    for item in index.values():
        subset = item.get("dataset_source")
        if not isinstance(subset, str) or not subset.strip():
            raise ValueError(f"Missing dataset_source for {item['video_path']}; cannot safely select a subset")
        if subset.strip().casefold() == "realestate10k":
            selected.append(item)
    if not selected:
        raise ValueError(f"No RealEstate10K entries in {source}")
    result = {"source": str(source), "source_sha256": original_hash, "input_entries": len(index),
              "selected_entries": len(selected), "excluded_entries": len(index) - len(selected),
              "output_file": Path(destination).name, "all_fields_preserved": True,
              "split_rule": "input archive; video_path train/test components are not used"}
    # Release other subsets before reloading the output for content verification.
    del index
    with Path(destination).open("xb") as handle:
        np.savez_compressed(handle, arr_0=np.asarray(selected, dtype=object))
    restored = list(load_official_camera_index(destination, data_root).values())
    if len(restored) != len(selected) or not all(metadata_equal(a, b) for a, b in zip(selected, restored)):
        raise ValueError(f"Output failed full-field roundtrip verification: {destination}")
    if sha256_file(source) != original_hash:
        raise ValueError(f"Source changed during extraction: {source}")
    result.update(output_sha256=sha256_file(destination), output_bytes=Path(destination).stat().st_size,
                  roundtrip_verified=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root")
    parser.add_argument("--data-root", default="public_data/RealCam-Vid/RealEstate10K", help="CSV/video root, relative to workspace")
    parser.add_argument("--source-root", default="public_data/RealCam-Vid", help="Original NPZ root, relative to workspace")
    parser.add_argument("--train-npz", default="RealCam-Vid_train.npz", help="Relative to source root, or absolute")
    parser.add_argument("--test-npz", default="RealCam-Vid_test.npz", help="Relative to source root, or absolute")
    parser.add_argument("--train-csv", help="Relative to data root, or absolute")
    parser.add_argument("--test-csv", help="Relative to data root, or absolute")
    parser.add_argument("--split-policy", choices=("csv", "archive"), default="csv",
                        help="csv: match CSV rows across both NPZs (default); archive: legacy subset filtering, original paths/splits")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), help="Default: both for extraction; test for --inspect-only")
    parser.add_argument("--inspect-only", action="store_true", help="Export all NPZ records as readable JSON instead of extracting a subset")
    parser.add_argument("--preview-count", type=int, default=3, help="Records printed as summaries and included in pretty preview.json; all records still exported")
    parser.add_argument("--run-name", help="Unique repository output subdirectory; no overwriting")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    args.splits = args.splits or (["test"] if args.inspect_only else ["train", "test"])
    if args.preview_count < 0:
        parser.error("--preview-count must be nonnegative")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits must not contain duplicates")
    root = workspace_root(args.workspace_root)
    data_root = resolve_path(args.data_root, root)
    source_root = resolve_path(args.source_root, root)
    csv_mode = not args.inspect_only and args.split_policy == "csv"
    prefix = "metadata_readable" if args.inspect_only else "metadata_realestate10k_csv" if csv_mode else "metadata_realestate10k"
    name = args.run_name or (f"{prefix}_{'-'.join(args.splits)}_{datetime.now():%Y%m%d-%H%M%S-%f}" +
                             (f"_{args.tag}" if args.tag else ""))
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", name) or name.endswith(".")
            or name.split(".")[0].upper() in reserved):
        raise ValueError("Run name must be a portable single folder name (max 180 characters)")
    folder = REPO_ROOT / "output" / name
    folder.mkdir(parents=True, exist_ok=False)
    inputs = {split: resolve_path(getattr(args, f"{split}_npz"), source_root)
              for split in (("train", "test") if csv_mode else args.splits)}
    parameters = {"arguments": vars(args), "workspace_root": str(root), "data_root": str(data_root),
                  "inputs": {key: str(value) for key, value in inputs.items()}, "output_folder": str(folder),
                  "source_root": str(source_root),
                  "mode": "readable_inspection" if args.inspect_only else "csv_partition" if csv_mode else "subset_extraction",
                  "subset": None if args.inspect_only else "RealEstate10K",
                  "filter_field": None if args.inspect_only else "dataset_source", "compressed": not args.inspect_only,
                  "verify_roundtrip": not args.inspect_only, "change_camera_coordinates": False,
                  "split_rule": "CSV membership" if csv_mode else "input archive", "normalize_video_path": csv_mode}
    write_json(folder / "parameters.json", parameters)
    write_json(folder / "launch.json", {"argv": sys.argv if argv is None else argv, "environment": environment_record()})
    (folder / "extraction.txt").write_text("RealCam metadata (step 3A): " + parameters["mode"] + "\n\n" +
                                         json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run directory: {folder}", flush=True)
    write_json(folder / "status.json", {"status": "running"})
    summaries = {}
    try:
        for path in inputs.values():
            if not path.is_file():
                raise FileNotFoundError(f"Input NPZ not found: {path}")
        if csv_mode:
            csv_paths = {split: discover_csv(data_root, split, getattr(args, f"{split}_csv"), subset="RealEstate10K") for split in args.splits}
            parameters["csv_inputs"] = {split: str(path) for split, path in csv_paths.items()}
            write_json(folder / "parameters.json", parameters)
            (folder / "extraction.txt").write_text("RealCam metadata (step 3A): csv_partition\n\n" +
                                                  json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries = partition_by_csv(inputs, csv_paths, data_root, source_root, folder, equal=metadata_equal)
            write_json(folder / "summary.json", summaries)
        for split, path in (() if csv_mode else inputs.items()):
            print(f"[{split}] reading {path.name}", flush=True)
            if args.inspect_only:
                summaries[split] = export_readable_npz(path, folder / split, source_root, preview_count=args.preview_count)
            else:
                summaries[split] = extract_split(path, folder / f"RealEstate10K_{split}.npz", source_root)
            write_json(folder / "summary.json", summaries)
            if not args.inspect_only:
                print(f"[{split}] selected {summaries[split]['selected_entries']}/{summaries[split]['input_entries']}; roundtrip verified", flush=True)
        write_json(folder / "status.json", {"status": "completed", "returncode": 0, "splits": args.splits})
        with (folder / "extraction.txt").open("a", encoding="utf-8") as handle:
            handle.write("\n\nResults:\n" + json.dumps(summaries, ensure_ascii=False, indent=2))
        if not args.inspect_only and set(args.splits) == {"train", "test"}:
            if csv_mode:
                command = ["python", "inspect_realcam.py", "--camera-metadata-dir", f"output/{name}", "--limit", "10", "--tag", "csv-aligned"]
                if root != workspace_root(repo_root=REPO_ROOT):
                    command += ["--workspace-root", portable_path(root, REPO_ROOT)]
                if data_root != root / "public_data/RealCam-Vid/RealEstate10K":
                    command += ["--set", f"paths.data_root={portable_path(data_root, root)}"]
                for split, path in csv_paths.items():
                    command += ["--set", f"data.{split}_csv={portable_path(path, data_root)}"]
                print_next_command(folder, command, REPO_ROOT)
            else:
                print("Legacy archive filtering keeps original prefixes/splits; use --split-policy csv for the new subset-root layout.", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        write_json(folder / "status.json", {"status": "failed", "returncode": 2,
                                          "error": str(exc), "completed_splits": list(summaries)})
        print(f"Extraction failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
