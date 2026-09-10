"""Temporary, read-only CSV/NPZ path diagnosis; remove after server validation.

No GPU or video decoding. Alternative matches are evidence only, never repairs.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from omegaconf import OmegaConf

from inspect_realcam import load_inspection_config
from utils.project_paths import REPO_ROOT, add_path_arguments, resolve_path
from utils.realcam_dataset import (csv_rows, data_path, discover_csv, field, load_official_camera_index,
                                  relative_path, resolve_columns, sha256_file)
from utils.realcam_inspection import identify_row
from utils.run_record import environment_record, write_json


def archive_evidence(path, root, rows):
    """Load each archive once, retain only matching path strings, then free arrays."""
    print(f"Reading NPZ: {path}", flush=True)
    before = sha256_file(path)
    index = load_official_camera_index(path, root)
    report = {"path": str(path), "sha256": before, "entries": len(index),
              "source_counts": dict(Counter(str(item.get("dataset_source", "<missing>")) for item in index.values())),
              "key_examples": list(index)[:3], "rows": []}
    for row in rows:
        key = row["canonical_video_path"]
        matches = {"exact": [], "case_only": [], "same_scene_and_file": [], "same_filename": []}
        counts = Counter()
        tail = "/".join(key.split("/")[-2:])
        for candidate, item in index.items():
            category = ("exact" if candidate == key else "case_only" if candidate.casefold() == key.casefold()
                        else "same_scene_and_file" if "/".join(candidate.split("/")[-2:]) == tail
                        else "same_filename" if candidate.split("/")[-1] == key.split("/")[-1] else None)
            if category:
                counts[category] += 1
                if len(matches[category]) < 5:
                    matches[category].append({"key": candidate, "raw_video_path": item["video_path"]})
        report["rows"].append({"split": row["split"], "video_path": key, "counts": dict(counts), "matches": matches})
    del index
    if sha256_file(path) != before:
        raise ValueError(f"NPZ changed while reading: {path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Default: audit resolved_config.yaml when supplied, otherwise inspection defaults")
    add_path_arguments(parser)
    parser.add_argument("--audit-dir", help="Previous inspection output, relative to repository")
    parser.add_argument("--camera-metadata-dir", help="Extracted NPZ directory, relative to repository")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--limit", type=int, default=10, help="Rows per split; previous rejected paths take priority")
    parser.add_argument("--compare-full", action="store_true", help="Also read available original full NPZs, even when all selected keys match")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    # Timestamp only: repeating this diagnostic never reuses the failed audit folder.
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", args.tag)[:60]
    folder = REPO_ROOT / "output" / (f"diagnose_realcam_temp_{datetime.now():%Y%m%d-%H%M%S-%f}" + (f"_{tag}" if tag else ""))
    folder.mkdir(parents=True, exist_ok=False)
    print(f"Diagnostic directory: {folder}", flush=True)
    write_json(folder / "parameters.json", {"arguments": vars(args), "argv": sys.argv if argv is None else argv,
                                            "environment": environment_record(), "output": str(folder)})
    report = {"scope": "CSV/NPZ path association and video file existence only; no video decoding or geometry validation",
              "read_only_inputs": True, "inputs": {}, "rows": [], "archives": [], "errors": []}
    write_json(folder / "status.json", {"status": "running"})
    try:
        audit = resolve_path(args.audit_dir, REPO_ROOT) if args.audit_dir else None
        if audit is not None and not audit.is_dir():
            raise FileNotFoundError(f"Audit directory not found: {audit}")
        source = args.config or (str(audit / "resolved_config.yaml") if audit else "configs/data/realcam_inspection.yaml")
        config = load_inspection_config(source, args.workspace_root, args.overrides)
        root = Path(config.paths.data_root)
        settings = OmegaConf.to_container(config.data, resolve=True)
        OmegaConf.save(config, folder / "resolved_config.yaml", resolve=True)
        prior_sources, rejected = {}, []
        if audit:
            for name in ("summary.json", "status.json", "metadata_sources.json"):
                path = audit / name
                if path.is_file():
                    content = json.loads(path.read_text(encoding="utf-8"))
                    write_json(folder / f"previous_{name}", content)
                    if name == "metadata_sources.json":
                        prior_sources = content
            path = audit / "rejected.csv"
            if path.is_file():
                iterator = csv_rows(path)
                headers = next(iterator)
                counts = Counter()
                for _, _, values in iterator:
                    row = dict(zip(headers, values))
                    split = row.get("original_split")
                    if counts[split] < args.limit:
                        rejected.append(row)
                        counts[split] += 1
                write_json(folder / "previous_rejected_examples.json", rejected)
        selected = {}
        for split in dict.fromkeys(args.splits):
            csv_path = discover_csv(root, split, settings.get(f"{split}_csv"), subset=settings["subset"])
            name = relative_path(csv_path, root)
            iterator = csv_rows(csv_path, encoding=settings["encoding"], delimiter=settings["delimiter"],
                                field_limit=config.inspection.csv_field_size_limit)
            headers = next(iterator)
            columns = resolve_columns(headers, settings["columns"], external_camera=True)
            prior = prior_sources.get(name, {}).get("official_camera_npz")
            camera = settings.get(f"{split}_camera_npz")
            if args.camera_metadata_dir:
                camera = resolve_path(args.camera_metadata_dir, REPO_ROOT) / f"{settings['subset']}_{split}.npz"
            elif not args.config and prior and not any(value.startswith(f"data.{split}_camera_npz=") for value in args.overrides):
                camera = prior["path"]
            if not camera:
                camera = next((root / f"{prefix}_{split}.npz" for prefix in (settings["subset"], "RealCam-Vid")
                               if (root / f"{prefix}_{split}.npz").is_file()), root / f"RealCam-Vid_{split}.npz")
            selected[split] = resolve_path(camera, root)
            wanted = {relative_path(data_path(row["video_path"], root), root)
                      for row in rejected if row.get("original_split") == split and row.get("video_path")}
            examples, failed_examples, seen = [], [], set()
            before = sha256_file(csv_path)
            for _, line, values in iterator:
                if len(values) != len(headers):
                    report["errors"].append(f"{name} row {line}: expected {len(headers)} columns, got {len(values)}")
                    continue
                # Reuse the production field mapping and subset/group identification.
                row = dict(zip(headers, values))
                identity = identify_row(row, columns, settings, split)
                if identity is None:
                    continue
                video = data_path(identity["video_path"], root)
                key = relative_path(video, root)
                sample = {"split": split, "csv_row": line, "raw_video_path": row.get(columns["video_path"], ""),
                          "canonical_video_path": key, "video_file": str(video), "video_exists": video.is_file(),
                          "selected_npz": str(selected[split]), "caption_present": bool(field(row, columns, "caption"))}
                if len(examples) < args.limit:
                    examples.append(sample)
                if key in wanted:
                    failed_examples.append(sample)
                    seen.add(key)
                if (not wanted and len(examples) >= args.limit) or (wanted and seen == wanted):
                    break
            if sha256_file(csv_path) != before:
                raise ValueError(f"CSV changed while reading: {csv_path}")
            chosen = failed_examples + [row for row in examples if row["canonical_video_path"] not in wanted]
            report["rows"] += chosen[:args.limit]
            report["inputs"][split] = {"csv": str(csv_path), "csv_sha256": before, "columns": columns,
                                       "selected_npz": str(selected[split]), "unseen_previous_paths": sorted(wanted - seen),
                                       "previous_csv_sha256": prior_sources.get(name, {}).get("sha256")}
            if wanted - seen:
                report["errors"].append(f"{split}: previous rejected paths no longer present in selected CSV: {sorted(wanted - seen)}")
            if not chosen:
                report["errors"].append(f"No selected CSV rows for {split}")
        checked = set()

        def inspect_archive(path):
            path = Path(path).resolve()
            if path in checked:
                return
            checked.add(path)
            if not path.is_file():
                report["errors"].append(f"NPZ not found: {path}")
                return
            try:
                report["archives"].append(archive_evidence(path, root, report["rows"]))
            except Exception as exc:
                report["errors"].append(f"NPZ {path}: {type(exc).__name__}: {exc}")
            write_json(folder / "report.json", report)

        for path in selected.values():
            inspect_archive(path)
        for index, row in enumerate(report["rows"]):
            row["exact_in_selected_npz"] = any(item["path"] == row["selected_npz"] and
                item["rows"][index]["counts"].get("exact", 0) == 1 for item in report["archives"])
        if args.compare_full or any(not row["exact_in_selected_npz"] for row in report["rows"]):
            directories = {root, *(path.parent for path in selected.values())}
            for directory in sorted(directories):
                for prefix in (settings["subset"], "RealCam-Vid"):
                    for split in ("train", "test"):
                        path = directory / f"{prefix}_{split}.npz"
                        if path.is_file():
                            inspect_archive(path)
        passed = not report["errors"] and bool(report["rows"]) and all(
            row["video_exists"] and row["caption_present"] and row["exact_in_selected_npz"] for row in report["rows"])
        report["passed"] = passed
        lines = [report["scope"]]
        for split, source in report["inputs"].items():
            lines.append(f"[{split}] CSV={source['csv']}; selected NPZ={source['selected_npz']}; columns={json.dumps(source['columns'])}")
        for archive in report["archives"]:
            lines.append(f"NPZ={archive['path']}; entries={archive['entries']}; sources={json.dumps(archive['source_counts'])}; "
                         f"key_examples={json.dumps(archive['key_examples'], ensure_ascii=False)}")
        for index, row in enumerate(report["rows"]):
            alternatives = [archive["path"] for archive in report["archives"]
                            if archive["path"] != row["selected_npz"] and archive["rows"][index]["counts"].get("exact")]
            row["exact_in_alternative_archives"] = alternatives
            lines.append(f"[{row['split']}] video_exists={row['video_exists']} selected_npz_exact={row['exact_in_selected_npz']} "
                         f"caption_present={row['caption_present']} video_path={row['canonical_video_path']}")
            if not row["exact_in_selected_npz"] and alternatives:
                lines.append("  Absent from selected NPZ; exact path found in alternative archives (diagnostic only): " +
                             json.dumps(alternatives, ensure_ascii=False))
            for archive in report["archives"]:
                evidence = archive["rows"][index]
                if evidence["counts"]:
                    lines.append(f"  {archive['path']}: {json.dumps(evidence['counts'])}")
                    for category, matches in evidence["matches"].items():
                        if matches:
                            lines.append(f"    {category}: {json.dumps(matches, ensure_ascii=False)}")
        lines += report["errors"]
        lines.append(f"Path diagnosis {'PASSED' if passed else 'FAILED'}; full stage-3 inspection is still required.")
        text = "\n".join(lines)
        (folder / "diagnosis.txt").write_text(text + "\n", encoding="utf-8")
        write_json(folder / "report.json", report)
        write_json(folder / "status.json", {"status": "passed" if passed else "mismatch", "returncode": 0 if passed else 1})
        print(text, flush=True)
        print(f"Share diagnosis.txt and report.json from: {folder}", flush=True)
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        write_json(folder / "report.json", report)
        (folder / "diagnosis.txt").write_text("\n".join(report["errors"]) + "\n", encoding="utf-8")
        write_json(folder / "status.json", {"status": "failed", "returncode": 2, "error": str(exc)})
        print(f"Diagnosis failed: {exc}\nReport: {folder}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
