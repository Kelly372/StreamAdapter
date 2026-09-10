"""Build portable, group-disjoint metadata manifests for the I2V pipeline."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from utils.project_paths import resolve_path

from utils.realcam_dataset import (
    MetadataError, camera_digest, contiguous_runs, csv_rows, data_path, discover_csv,
    field, load_camera, load_official_camera_index, numeric_field, probe_video, relative_path, resolve_columns,
    sha256_file, source_partition,
)


MANIFEST_FIELDS = [
    "sample_id", "clip_id", "source_id", "group_id", "group_basis", "subset", "original_split", "split", "video_path",
    "caption", "short_caption", "align_factor", "camera_scale", "vtss_score",
    "metadata_csv", "metadata_row", "metadata_offset", "camera_sha256", "camera_cache",
    "camera_frames", "valid_camera_frames", "valid_runs", "num_frames", "fps", "width", "height",
    "max_valid_span_seconds", "eligible_target_window", "verification",
]
ISSUE_FIELDS = ["original_split", "metadata_csv", "metadata_row", "video_path", "source_id", "group_id", "reason", "detail"]


def write_csv(path, records, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def identify_row(row, columns, settings, original_split):
    video = field(row, columns, "video_path")
    subset = field(row, columns, "subset")
    if not subset:
        known = {"realestate10k": "RealEstate10K", "dl3dv": "DL3DV", "dl3dv10k": "DL3DV", "miradata": "MiraData"}
        subset = next((known[_name(part)] for part in video.replace("\\", "/").split("/") if _name(part) in known), "")
    subset = subset or settings.get("assume_subset")
    if not subset:
        raise MetadataError("unknown_subset", "Provide a subset column, path component, or explicit data.assume_subset")
    if _name(subset) != _name(settings["subset"]):
        return None
    split = field(row, columns, "split")
    if split and split.lower() != original_split:
        raise MetadataError("split_mismatch", f"Row says {split}, file selected as {original_split}")
    source_id = field(row, columns, "source_id")
    if not source_id and settings.get("source_id_regex"):
        match = re.search(settings["source_id_regex"], video.replace("\\", "/"))
        if match:
            source_id = match.groupdict().get("source_id") or (match.group(1) if match.lastindex else "")
    if not source_id and settings["group_by"] == "source_id":
        raise MetadataError("missing_source_id", "Map an original-video ID column or configure data.source_id_regex; clip ID is not a safe substitute")
    if source_id:
        group_id, group_basis = f"source:{source_id}", "source_id"
    else:
        parts = video.replace("\\", "/").split("/")
        # Only the confirmed RealEstate10K/<original split>/<group>/<clip> layout.
        # Original data folder split is NOT the RealCam-Vid metadata split.
        matches = [i for i, part in enumerate(parts) if _name(part) == _name(settings["subset"])]
        if not matches or len(parts) - matches[-1] != 4 or parts[matches[-1] + 1] not in ("train", "test"):
            raise MetadataError("unknown_directory_group", "Expected RealEstate10K/{train|test}/<group>/<video>; otherwise provide source_id")
        parent = parts[-2]
        if parent in ("", ".", ".."):
            raise MetadataError("unknown_directory_group", "Empty or relative parent directory")
        group_id, group_basis = f"directory:{settings['subset']}/{parent}", "parent_directory"
    return {"source_id": source_id, "group_id": group_id, "group_basis": group_basis,
            "clip_id": field(row, columns, "clip_id") or Path(video.replace("\\", "/")).stem,
            "video_path": video, "subset": settings["subset"]}


def inspect_row(row, columns, identity, root, settings, checks, probe=probe_video, external_payload=None, camera_cache=None):
    path = data_path(identity["video_path"], root)
    if not path.is_file():
        raise MetadataError("missing_video", str(path))
    caption = field(row, columns, "caption")
    if checks["require_caption"] and not caption:
        raise MetadataError("missing_caption", "A nonempty long_caption/caption is required")
    camera = load_camera(row, columns, root, rotation_tolerance=checks["rotation_tolerance"],
                         timestamps_unit=settings["timestamps_unit"], external_payload=external_payload)
    frame_count = len(camera["camera_extrinsics"])
    valid = camera["valid_mask"]
    if not valid.any() or float(valid.mean()) < checks["min_valid_pose_fraction"]:
        raise MetadataError("invalid_camera_fraction", f"{int(valid.sum())}/{frame_count} valid camera frames")
    video = {key: numeric_field(row, columns, key, integer=key != "fps")
             for key in ("fps", "num_frames", "width", "height")}
    video.update(timestamps=None, verification="metadata_only")
    if checks["probe_mode"] != "metadata":
        try:
            observed = probe(path, checks["probe_mode"])
        except MetadataError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise MetadataError("video_decode_error", str(exc)) from exc
        for key in ("num_frames", "width", "height"):
            if video[key] is not None and video[key] != observed[key]:
                raise MetadataError("video_metadata_mismatch", f"{key}: CSV={video[key]}, video={observed[key]}")
        if video["fps"] is not None and abs(video["fps"] - observed["fps"]) / observed["fps"] > checks["fps_relative_tolerance"]:
            raise MetadataError("fps_mismatch", f"CSV={video['fps']}, video={observed['fps']}")
        video.update(observed)
    if video["num_frames"] is not None and video["num_frames"] != frame_count:
        raise MetadataError("camera_video_length_mismatch", f"camera={frame_count}, video={video['num_frames']}")
    camera_times, video_times = camera["timestamps"], video["timestamps"]
    if camera_times is not None and video_times is not None:
        error = np.max(np.abs((camera_times - camera_times[0]) - (video_times - video_times[0])))
        if error > checks["timestamp_tolerance_seconds"]:
            raise MetadataError("camera_video_timestamp_mismatch", f"max_error_seconds={error}")
    times = video_times if video_times is not None else camera_times
    if times is None and video["fps"]:
        times = np.arange(frame_count, dtype=np.float64) / video["fps"]
    runs = contiguous_runs(valid)
    span = max(float(times[end - 1] - times[start]) for start, end in runs) if times is not None else None
    required_span = (settings["rgb_frames"] - 1) / settings["target_fps"]
    # This is a duration eligibility check, NOT synchronized window sampling.
    eligible = "unknown" if span is None else "yes" if span + 1e-6 >= required_span else "no"
    result = dict(identity)
    for key in ("camera_scale", "vtss_score"):
        raw = field(row, columns, key)
        value = float(raw) if raw else None
        if value is not None and (not math.isfinite(value) or (key == "camera_scale" and value < 0)):
            raise MetadataError("invalid_quality_metadata", f"{key}={raw!r}")
        result[key] = value
    result.update(video_path=relative_path(path, root), caption=caption,
                  short_caption=field(row, columns, "short_caption"), align_factor=camera["align_factor"],
                  camera_sha256=camera_digest(camera), camera_frames=frame_count,
                  valid_camera_frames=int(valid.sum()), valid_runs=json.dumps(runs),
                  max_valid_span_seconds=span, eligible_target_window=eligible,
                  **{key: video[key] for key in ("num_frames", "fps", "width", "height", "verification")})
    result["sample_id"] = hashlib.sha256(result["video_path"].encode("utf-8")).hexdigest()[:24]
    if camera_cache is not None:
        cache_path = Path(camera_cache) / f"{result['sample_id']}.npz"
        np.savez_compressed(cache_path, **{key: value for key, value in camera.items() if value is not None})
        result["camera_cache"] = f"cameras/{cache_path.name}"
    return result


def inspect_dataset(config, output_dir, probe=probe_video):
    from omegaconf import OmegaConf

    settings = OmegaConf.to_container(config.data, resolve=True)
    checks = OmegaConf.to_container(config.inspection, resolve=True)
    root = Path(config.paths.data_root).resolve()
    output = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if checks["probe_mode"] not in ("metadata", "header", "decode"):
        raise ValueError("inspection.probe_mode must be metadata, header or decode")
    if not 0 <= settings["validation_fraction"] < 1 or not 0 <= checks["min_valid_pose_fraction"] <= 1:
        raise ValueError("Invalid validation_fraction or min_valid_pose_fraction")
    if checks["limit_per_split"] < 0 or settings["target_fps"] <= 0 or settings["rgb_frames"] < 2:
        raise ValueError("Invalid inspection limit, target_fps or rgb_frames")
    if any(not math.isfinite(checks[key]) or checks[key] < 0 for key in (
            "rotation_tolerance", "fps_relative_tolerance", "timestamp_tolerance_seconds")):
        raise ValueError("Inspection tolerances must be finite and nonnegative")
    if checks["csv_field_size_limit"] <= 0 or checks["progress_every"] < 0:
        raise ValueError("Invalid CSV field size or progress interval")
    if settings["timestamps_unit"] not in ("seconds", "milliseconds", "microseconds"):
        raise ValueError("Specify timestamps_unit as seconds, milliseconds or microseconds")
    if settings["camera_convention"] not in ("opencv_w2c", "opencv_c2w"):
        raise ValueError("Camera convention must be explicit: opencv_w2c or opencv_c2w")
    if settings["intrinsics_units"] not in ("normalized", "pixels"):
        raise ValueError("intrinsics_units must be normalized or pixels")
    if settings["group_by"] not in ("source_id", "source_or_directory"):
        raise ValueError("group_by must be source_id or source_or_directory")
    if settings.get("source_id_regex"):
        pattern = re.compile(settings["source_id_regex"])
        if pattern.groups < 1:
            raise ValueError("source_id_regex must capture the original-video ID")
    if checks["probe_mode"] != "metadata" and probe is probe_video:
        try:
            import av  # noqa: F401 - fail once before scanning a large dataset
        except ImportError as exc:
            raise RuntimeError("Install PyAV for video checks: python -m pip install av==13.1.0") from exc

    candidates, rejects = [], []
    source_splits = defaultdict(set)
    group_splits, group_bases = defaultdict(set), Counter()
    path_counts = Counter()
    stats = {split: Counter() for split in ("train", "test")}
    metadata_sources = {}
    for split in ("train", "test"):
        csv_path = discover_csv(root, split, settings.get(f"{split}_csv"), subset=settings["subset"])
        relative_csv = relative_path(csv_path, root)
        print(f"[{split}] reading CSV: {relative_csv}", flush=True)
        before_hash = sha256_file(csv_path)
        iterator = csv_rows(csv_path, encoding=settings["encoding"], delimiter=settings["delimiter"],
                            field_limit=checks["csv_field_size_limit"])
        headers = next(iterator)
        camera_npz = settings.get(f"{split}_camera_npz")
        mapped = resolve_columns(headers, settings["columns"], external_camera=True)
        if not camera_npz and not mapped["camera_path"] and not (mapped["intrinsics"] and mapped["extrinsics"]):
            known = [root / f"{prefix}_{split}.npz" for prefix in (settings["subset"], "RealCam-Vid")]
            available = list(dict.fromkeys(path for path in known if path.is_file()))
            # A dedicated subset archive wins over the full archive when both exist.
            camera_npz = available[0] if available else None
        camera_index, camera_source = None, None
        if camera_npz:
            camera_npz = resolve_path(camera_npz, root)
            camera_hash = sha256_file(camera_npz)
            print(f"[{split}] loading camera metadata: {camera_npz.name}", flush=True)
            camera_index = load_official_camera_index(camera_npz, root)
            camera_source = {"path": relative_path(camera_npz, root) if camera_npz.is_relative_to(root) else str(camera_npz),
                             "path_base": "data_root" if camera_npz.is_relative_to(root) else "absolute",
                             "sha256": camera_hash}
            (output / "cameras").mkdir(exist_ok=True)
        columns = resolve_columns(headers, settings["columns"], external_camera=camera_index is not None)
        metadata_sources[relative_csv] = {
            "sha256": before_hash, "headers": headers, "columns": columns,
            "encoding": settings["encoding"], "delimiter": settings["delimiter"],
            "field_size_limit": checks["csv_field_size_limit"],
            "rotation_tolerance": checks["rotation_tolerance"],
            "timestamps_unit": settings["timestamps_unit"], "camera_convention": settings["camera_convention"],
            "intrinsics_units": settings["intrinsics_units"],
            "official_camera_npz": camera_source,
            "video_path_base": "data_root",
            "camera_match_key": "normalized data-root-relative video_path",
        }
        for offset, row_number, values in iterator:
            stats[split]["rows_total"] += 1
            issue = {"original_split": split, "metadata_csv": relative_csv, "metadata_row": row_number}
            try:
                if len(values) != len(headers):
                    raise MetadataError("csv_column_count", f"Expected {len(headers)} fields, got {len(values)}")
                row = dict(zip(headers, values))
                issue["video_path"] = field(row, columns, "video_path")
                identity = identify_row(row, columns, settings, split)
                if identity is None:
                    stats[split]["other_subsets"] += 1
                    continue
                issue["source_id"] = identity["source_id"]
                issue["group_id"] = identity["group_id"]
                if identity["source_id"]:
                    source_splits[identity["source_id"]].add(split)
                group_splits[identity["group_id"]].add(split)
                group_bases[identity["group_basis"]] += 1
                canonical_path = relative_path(data_path(identity["video_path"], root), root)
                path_counts[canonical_path.casefold()] += 1
                stats[split]["selected_subset"] += 1
                limit = checks["limit_per_split"]
                if limit and stats[split]["inspected"] >= limit:
                    stats[split]["not_inspected_limit"] += 1
                    continue
                stats[split]["inspected"] += 1
                external_payload = None
                if camera_index is not None:
                    external_payload = camera_index.get(canonical_path)
                    if external_payload is None:
                        raise MetadataError("missing_camera_metadata",
                            f"No NPZ entry matches video_path={canonical_path}; "
                            f"CSV={relative_csv}; camera_npz={camera_source['path']}; "
                            f"video_file={data_path(canonical_path, root)}; "
                            "matching uses the complete data-root-relative path, preserving train/test directories")
                result = inspect_row(row, columns, identity, root, settings, checks, probe=probe,
                                     external_payload=external_payload,
                                     camera_cache=output / "cameras" if camera_index is not None else None)
                result.update(original_split=split, metadata_csv=relative_csv,
                              metadata_row=row_number, metadata_offset=offset)
                candidates.append(result)
            except MetadataError as exc:
                rejects.append(dict(issue, reason=exc.code, detail=str(exc)))
            except (OSError, ValueError, TypeError) as exc:
                rejects.append(dict(issue, reason="invalid_row_payload", detail=str(exc)))
            if checks["progress_every"] and stats[split]["rows_total"] % checks["progress_every"] == 0:
                print(f"[{split}] scanned={stats[split]['rows_total']}, inspected={stats[split]['inspected']}", flush=True)
        if before_hash != sha256_file(csv_path):
            raise RuntimeError(f"CSV changed during inspection: {csv_path}; retry with a stable input")
        if camera_source and camera_source["sha256"] != sha256_file(camera_npz):
            raise RuntimeError(f"Camera NPZ changed during inspection: {camera_npz}")
        # Release the full object archive before loading the next split.
        del camera_index

    overlaps = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    group_overlaps = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    partitions = {name: [] for name in ("train", "validation", "test")}
    for record in candidates:
        reason = ("source_split_overlap" if record["source_id"] else "directory_group_split_overlap") if len(group_splits[record["group_id"]]) > 1 else (
            "duplicate_video" if path_counts[record["video_path"].casefold()] > 1 else None)
        if reason:
            rejects.append(dict(record, reason=reason, detail="Excluded all affected rows; original splits are never merged"))
            continue
        split = "test" if record["original_split"] == "test" else source_partition(
            record["source_id"] or record["group_id"], config.seed, settings["validation_fraction"])
        record["split"] = split
        partitions[split].append(record)
    for split, records in partitions.items():
        write_csv(output / f"{split}.csv", records, MANIFEST_FIELDS)
    write_csv(output / "rejected.csv", rejects, ISSUE_FIELDS)
    (output / "metadata_sources.json").write_text(json.dumps(metadata_sources, ensure_ascii=False, indent=2), encoding="utf-8")
    all_records = [item for records in partitions.values() for item in records]
    summary = {
        "schema_version": 1, "subset": settings["subset"], "probe_mode": checks["probe_mode"],
        "camera_coordinates_transformed": False, "windows_sampled": False,
        "all_selected_rows_inspected": not any(stats[s]["not_inspected_limit"] for s in stats),
        "scan": {split: dict(counts) for split, counts in stats.items()},
        "accepted": {split: len(records) for split, records in partitions.items()},
        "groups": {split: len({r['group_id'] for r in records}) for split, records in partitions.items()},
        "grouping_bases_in_scanned_rows": dict(group_bases),
        "original_source_leakage_check_complete": bool(group_bases) and not group_bases["parent_directory"] and not any(
            r["reason"] in ("missing_source_id", "unknown_subset", "unknown_directory_group", "csv_column_count", "invalid_row_payload")
            for r in rejects),
        "rejected": len(rejects), "rejection_reasons": dict(Counter(item["reason"] for item in rejects)),
        "overlapping_train_test_sources": overlaps,
        "overlapping_train_test_groups": group_overlaps,
        "duration_eligibility": dict(Counter(item["eligible_target_window"] for item in all_records)),
        "verification_levels": dict(Counter(item["verification"] for item in all_records)),
        "target_fps": settings["target_fps"], "rgb_frames": settings["rgb_frames"],
        "note": "Duration eligibility does not establish synchronized frame sampling or detect shot cuts. See step 4.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
