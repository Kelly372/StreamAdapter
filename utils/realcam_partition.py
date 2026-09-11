"""Build CSV-aligned subset archives using both original camera archives."""
import pickle
from pathlib import Path
import sqlite3

import numpy as np

from utils.realcam_dataset import (csv_rows, data_path, field, load_official_camera_index,
                                  relative_path, resolve_columns, sha256_file)
from utils.run_record import write_json


def subset_video_path(value, root, *, original=False):
    """Normalize separators and ./, but never rewrite train/test or guess filenames."""
    raw = str(value).strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"Parent traversal in video_path: {value!r}")
    if original and parts and parts[0] == "RealEstate10K":
        raw = "/".join(parts[1:])
    path = data_path(raw, root)
    key = relative_path(path, root)
    parts = key.split("/")
    if len(parts) != 3 or parts[0] not in ("train", "test"):
        raise ValueError(f"Expected train/<scene>/<video> or test/<scene>/<video> relative to RealEstate10K: {value!r}")
    return key


def partition_by_csv(inputs, csv_paths, data_root, source_root, output, *, equal):
    """Fail closed on missing/conflicting records; never edit input CSVs or NPZs.

    SQLite stages selected records so arrays from both full archives never need
    to remain resident together. Output archives are materialized one at a time.
    """
    report = {"split_rule": "input CSV membership and row order", "data_root": str(data_root),
              "source_root": str(source_root), "csvs": {}, "sources": {}, "issues": [], "splits": {},
              "normalization": "remove original RealEstate10K/ prefix only; preserve train/test path component",
              "video_decode_checked": False, "camera_geometry_checked": False}
    rows, owners = {}, {}
    for split, csv_path in csv_paths.items():
        before = sha256_file(csv_path)
        iterator = csv_rows(csv_path)
        headers = next(iterator)
        columns = resolve_columns(headers, external_camera=True)
        rows[split] = []
        for _, line, values in iterator:
            try:
                if len(values) != len(headers):
                    raise ValueError(f"Expected {len(headers)} columns, got {len(values)}")
                row = dict(zip(headers, values))
                if field(row, columns, "subset").casefold() != "realestate10k":
                    raise ValueError("CSV dataset_source must be RealEstate10K")
                key = subset_video_path(field(row, columns, "video_path"), data_root)
                if key in owners:
                    raise ValueError(f"Duplicate video_path (possibly cross-split): {key}; first seen in {owners[key]}")
                owners[key] = split
                rows[split].append({"video_path": key, "csv_row": line})
            except ValueError as exc:
                report["issues"].append({"split": split, "csv_row": line, "reason": "invalid_csv_row", "detail": str(exc)})
        if not rows[split]:
            report["issues"].append({"split": split, "reason": "empty_csv_split"})
        if sha256_file(csv_path) != before:
            raise ValueError(f"CSV changed during scan: {csv_path}")
        report["csvs"][split] = {"path": str(csv_path), "sha256": before, "rows": len(rows[split])}
    write_json(output / "partition_report.json", report)
    if report["issues"]:
        raise ValueError("CSV validation failed; see partition_report.json")
    stage = output / "_partition_staging.sqlite3"
    connection = sqlite3.connect(stage)
    try:
        connection.execute("CREATE TABLE records (path TEXT PRIMARY KEY, payload BLOB, sources TEXT)")
        for source_split, source in inputs.items():
            print(f"[{source_split}] indexing camera archive: {source}", flush=True)
            before = sha256_file(source)
            archive = load_official_camera_index(source, source_root)
            counts = {"entries": len(archive), "realestate10k": 0, "matched_csv": 0, "identical_duplicates": 0}
            for item in archive.values():
                if str(item.get("dataset_source", "")).strip().casefold() != "realestate10k":
                    continue
                counts["realestate10k"] += 1
                key = subset_video_path(item["video_path"], data_root, original=True)
                if key not in owners:
                    continue
                counts["matched_csv"] += 1
                normalized = dict(item, video_path=key)
                previous = connection.execute("SELECT payload, sources FROM records WHERE path=?", (key,)).fetchone()
                if previous:
                    # These bytes were serialized locally from the restricted NPZ loader.
                    if not equal(pickle.loads(previous[0]), normalized):
                        report["issues"].append({"reason": "conflicting_camera_metadata", "video_path": key,
                                                 "archives": previous[1].split(",") + [source_split]})
                    else:
                        counts["identical_duplicates"] += 1
                        connection.execute("UPDATE records SET sources=? WHERE path=?", (previous[1] + "," + source_split, key))
                else:
                    connection.execute("INSERT INTO records VALUES (?, ?, ?)",
                                       (key, pickle.dumps(normalized, protocol=4), source_split))
            del archive
            connection.commit()
            if sha256_file(source) != before:
                raise ValueError(f"Source changed during indexing: {source}")
            report["sources"][source_split] = {"path": str(source), "sha256": before, **counts}
            write_json(output / "partition_report.json", report)
        for split, entries in rows.items():
            for entry in entries:
                if not connection.execute("SELECT 1 FROM records WHERE path=?", (entry["video_path"],)).fetchone():
                    report["issues"].append({"split": split, **entry, "reason": "missing_camera_metadata"})
        write_json(output / "partition_report.json", report)
        if report["issues"]:
            raise ValueError("Missing or conflicting camera metadata; see partition_report.json. No subset NPZs published.")
        for split, entries in rows.items():
            selected = []
            provenance = []
            for entry in entries:
                payload, sources = connection.execute("SELECT payload, sources FROM records WHERE path=?", (entry["video_path"],)).fetchone()
                selected.append(pickle.loads(payload))
                provenance.append({**entry, "source_archives": sources.split(",")})
            destination = output / f"RealEstate10K_{split}.npz"
            with destination.open("xb") as handle:
                np.savez_compressed(handle, arr_0=np.asarray(selected, dtype=object))
            # Reload with the same path-keyed reader used by stage 3.
            restored = load_official_camera_index(destination, data_root)
            if list(restored) != [entry["video_path"] for entry in entries] or not all(
                    equal(original, reloaded) for original, reloaded in zip(selected, restored.values())):
                raise ValueError(f"CSV order/path or all-field roundtrip verification failed: {destination}")
            del selected, restored
            write_json(output / f"{split}_provenance.json", provenance)
            report["splits"][split] = {"selected_entries": len(entries), "output_file": destination.name,
                                        "output_sha256": sha256_file(destination), "roundtrip_verified": True,
                                        "csv_paths_match_exactly": True}
            print(f"[{split}] CSV-aligned entries: {len(entries)}; all-field roundtrip verified", flush=True)
        for split, csv_path in csv_paths.items():
            if sha256_file(csv_path) != report["csvs"][split]["sha256"]:
                raise ValueError(f"CSV changed during partitioning: {csv_path}")
        for split, source in inputs.items():
            if sha256_file(source) != report["sources"][split]["sha256"]:
                raise ValueError(f"NPZ changed during partitioning: {source}")
        write_json(output / "partition_report.json", report)
        return report["splits"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "partition_report.json", report)
        raise
    finally:
        connection.close()
        stage.unlink(missing_ok=True)
