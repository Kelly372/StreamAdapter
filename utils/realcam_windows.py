"""Step 4: timestamp-based I2V windows and synchronized pinhole intrinsics.

No Torch/model dependency. Camera extrinsics and align_factor stay in their
input convention; window-relative coordinates and Pluecker rays are step 5.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from utils.project_paths import REPO_ROOT, resolve_path
from utils.realcam_dataset import MetadataError, RealCamMetadataDataset, data_path, sha256_file


def validate_settings(settings):
    for name in ("rgb_frames", "height", "width"):
        if not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings["rgb_frames"] < 2:
        raise ValueError("rgb_frames must be at least 2")
    for name in ("target_fps", "max_frame_gap_seconds"):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("max_time_error_seconds", "camera_time_tolerance_seconds"):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if settings["resize_mode"] not in ("cover", "stretch") or settings["crop_mode"] not in ("center", "random"):
        raise ValueError("resize_mode: cover/stretch; crop_mode: center/random")
    if settings["shot_mode"] not in ("detect", "annotated"):
        raise ValueError("shot_mode must be detect or annotated")
    for name in ("cut_mean_threshold", "cut_hist_threshold", "cut_min_mean_for_hist"):
        if not math.isfinite(settings[name]) or not 0 < settings[name] <= 1:
            raise ValueError(f"{name} must be in (0,1]")
    if not isinstance(settings["cut_thumbnail_size"], int) or settings["cut_thumbnail_size"] < 4:
        raise ValueError("cut_thumbnail_size must be an integer >=4")
    if not isinstance(settings["cut_hist_bins"], int) or settings["cut_hist_bins"] < 2:
        raise ValueError("cut_hist_bins must be an integer >=2")


def frame_signature(rgb, bins):
    values = np.asarray(rgb, dtype=np.float32) / 255.
    histogram = np.concatenate([np.histogram(values[..., c], bins=bins, range=(0, 1))[0]
                                for c in range(3)]).astype(np.float64)
    histogram /= histogram.sum()
    return values, histogram


def scan_video(path, settings):
    """Retain timestamps and cut scores only; do not retain full video pixels."""
    import av
    times, cuts, scores = [], [], []
    previous = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise MetadataError("no_video_stream", str(path))
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (width, height):
                raise MetadataError("variable_resolution", str(path))
            if frame.time is None:
                raise MetadataError("missing_video_timestamps", str(path))
            times.append(float(frame.time))
            if settings["shot_mode"] == "detect":
                size = settings["cut_thumbnail_size"]
                thumb = frame.reformat(width=size, height=size, format="rgb24").to_ndarray()
                current = frame_signature(thumb, settings["cut_hist_bins"])
                mean, histogram = 0., 0.
                if previous is not None:
                    mean = float(np.abs(current[0] - previous[0]).mean())
                    histogram = float(np.abs(current[1] - previous[1]).sum() / 2)
                    if mean >= settings["cut_mean_threshold"] or (histogram >= settings["cut_hist_threshold"] and mean >= settings["cut_min_mean_for_hist"]):
                        cuts.append(len(times) - 1)
                scores.append((mean, histogram))
                previous = current
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise MetadataError("invalid_video_timestamps", str(path))
    return {"timestamps": times, "detected_cuts": cuts, "cut_scores": np.asarray(scores, dtype=np.float32).reshape(-1, 2),
            "width": width, "height": height, "fps": fps}


def validate_cuts(cuts, frames):
    if not isinstance(cuts, list) or any(type(value) is not int or not 0 < value < frames for value in cuts):
        raise MetadataError("invalid_shot_boundaries", "Cuts must be a list of frame indices in [1,F-1], each the first frame of a new shot")
    return sorted(set(cuts))


def valid_segments(timestamps, valid_mask, cuts, max_gap):
    """Half-open segments; never bridge invalid poses, cuts or large PTS gaps."""
    breaks = set(cuts) | set((np.flatnonzero(np.diff(timestamps) > max_gap + 1e-9) + 1).tolist())
    segments, start = [], None
    for i, valid in enumerate(valid_mask):
        if start is not None and (not valid or i in breaks):
            segments.append((start, i))
            start = None
        if valid and start is None:
            start = i
    if start is not None:
        segments.append((start, len(valid_mask)))
    return segments


def nearest_indices(timestamps, targets):
    upper = np.clip(np.searchsorted(timestamps, targets, side="left"), 0, len(timestamps) - 1)
    lower = np.maximum(upper - 1, 0)
    # Stable earlier-frame tie break; same indices are later used for cameras.
    return np.where(np.abs(timestamps[lower] - targets) <= np.abs(timestamps[upper] - targets) + 1e-12, lower, upper)


def window_indices(timestamps, start, settings, *, end=None):
    end = len(timestamps) if end is None else end
    targets = timestamps[start] + np.arange(settings["rgb_frames"], dtype=np.float64) / settings["target_fps"]
    if end < len(timestamps) and targets[-1] >= timestamps[end] - 1e-9:
        return None
    indices = nearest_indices(timestamps[start:end], targets) + start
    errors = np.abs(timestamps[indices] - targets)
    if not (np.diff(indices) > 0).all() or errors.max() > settings["max_time_error_seconds"] + 1e-9:
        return None
    return indices, targets


def feasible_starts(timestamps, valid_mask, cuts, settings):
    if np.asarray(valid_mask).shape != np.asarray(timestamps).shape:
        raise MetadataError("camera_video_length_mismatch", "Timestamp and valid-mask lengths differ")
    segments = valid_segments(timestamps, valid_mask, cuts, settings["max_frame_gap_seconds"])
    starts = []
    duration = (settings["rgb_frames"] - 1) / settings["target_fps"]
    for begin, end in segments:
        # A unique-source-frame window requires at least rgb_frames frames.
        for start in range(begin, end - settings["rgb_frames"] + 1):
            if timestamps[end - 1] - timestamps[start] + settings["max_time_error_seconds"] + 1e-9 < duration:
                break
            if window_indices(timestamps, start, settings, end=end) is not None:
                starts.append(start)
    return np.asarray(starts, dtype=np.int64), segments


def explain_window_rejection(timestamps, valid_mask, cuts, settings, segments):
    """Report failed constraints without relaxing production window selection."""
    times = np.asarray(timestamps)
    mask = np.asarray(valid_mask, dtype=bool)
    required = settings["rgb_frames"]
    duration = (required - 1) / settings["target_fps"]
    tolerance = settings["max_time_error_seconds"] + 1e-9
    lengths = [end - begin for begin, end in segments]
    spans = [float(times[end - 1] - times[begin]) for begin, end in segments]
    gap_indices = (np.flatnonzero(np.diff(times) > settings["max_frame_gap_seconds"] + 1e-9) + 1).tolist()
    result = {"required_frames": required, "target_fps": settings["target_fps"],
              "required_timestamp_span_seconds": duration, "source_frames": len(times),
              "source_timestamp_span_seconds": float(times[-1] - times[0]),
              "source_average_fps_from_timestamps": float((len(times) - 1) / (times[-1] - times[0])),
              "max_time_error_seconds": settings["max_time_error_seconds"],
              "max_frame_gap_seconds": settings["max_frame_gap_seconds"],
              "cut_count": len(cuts), "cut_frame_indices": list(cuts),
              "gap_count": len(gap_indices), "gap_frame_indices": gap_indices,
              "invalid_pose_frames": int((~mask).sum()), "segment_count": len(segments),
              "longest_segment_frames": max(lengths, default=0),
              "longest_segment_span_seconds": max(spans, default=0.),
              "segments_meeting_frame_and_span_requirements": sum(
                  n >= required and span + tolerance >= duration for n, span in zip(lengths, spans)),
              "causes": [], "diagnostic_only": True}
    if len(times) < required:
        result["causes"].append("too_few_source_frames")
    if times[-1] - times[0] + tolerance < duration:
        result["causes"].append("insufficient_source_duration")
    if result["causes"]:
        return result
    # These counterfactuals describe which restrictions block a window; they never
    # provide indices to the actual sampler or label detected cuts as incorrect.
    no_gaps = dict(settings, max_frame_gap_seconds=float("inf"))
    all_valid = np.ones_like(mask)
    raw_possible = bool(len(feasible_starts(times, all_valid, [], no_gaps)[0]))
    result["possible_without_pose_cut_gap_constraints"] = raw_possible
    if not raw_possible:
        result["causes"].append("unique_frame_or_timestamp_sampling")
        return result
    trials = (("shot_boundaries", mask, [], settings, bool(cuts)),
              ("timestamp_gaps", mask, cuts, no_gaps, bool(gap_indices)),
              ("invalid_poses", all_valid, cuts, settings, not mask.all()))
    for cause, trial_mask, trial_cuts, trial_settings, relevant in trials:
        possible = bool(len(feasible_starts(times, trial_mask, trial_cuts, trial_settings)[0])) if relevant else False
        result[f"possible_without_{cause}"] = possible
        if possible:
            result["causes"].append(cause)
    if not result["causes"]:
        result["causes"].append("combined_pose_cut_gap_constraints")
    return result


def spatial_transform(width, height, settings, rng):
    target_w, target_h = settings["width"], settings["height"]
    if settings["resize_mode"] == "stretch":
        resized_w, resized_h = target_w, target_h
    else:
        scale = max(target_w / width, target_h / height)
        resized_w, resized_h = max(target_w, math.ceil(width * scale)), max(target_h, math.ceil(height * scale))
    if settings["crop_mode"] == "random":
        left, top = int(rng.integers(resized_w - target_w + 1)), int(rng.integers(resized_h - target_h + 1))
    else:
        left, top = (resized_w - target_w) // 2, (resized_h - target_h) // 2
    sx, sy = resized_w / width, resized_h / height
    # Pixel centers use integer coordinates (top-left center is 0,0).
    # Pillow's half-pixel resize: u'=(u+.5)*sx-.5-left, likewise v.
    matrix = np.array([[sx, 0, (sx - 1) / 2 - left], [0, sy, (sy - 1) / 2 - top], [0, 0, 1]], dtype=np.float64)
    return {"source_width": width, "source_height": height, "resized_width": resized_w, "resized_height": resized_h,
            "crop_left": left, "crop_top": top, "output_width": target_w, "output_height": target_h,
            "pixel_transform": matrix.tolist(), "pixel_center_convention": "integer", "resampler": "pillow_bilinear"}


def transform_intrinsics(intrinsics, units, spatial):
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 4:
        matrices = np.zeros((len(values), 3, 3), dtype=np.float64)
        matrices[:, 0, 0], matrices[:, 1, 1] = values[:, 0], values[:, 1]
        matrices[:, 0, 2], matrices[:, 1, 2], matrices[:, 2, 2] = values[:, 2], values[:, 3], 1
    elif values.ndim == 3 and values.shape[1:] == (3, 3):
        matrices = values.copy()
    else:
        raise ValueError("Expected per-frame intrinsics [F,4] or [F,3,3]")
    if units == "normalized":
        matrices[:, 0, :] *= spatial["source_width"]
        matrices[:, 1, :] *= spatial["source_height"]
    elif units != "pixels":
        raise ValueError("Unknown intrinsics units")
    return np.asarray(spatial["pixel_transform"]) @ matrices


def decode_window(path, indices, expected_times, spatial):
    import av
    height, width = spatial["output_height"], spatial["output_width"]
    video = np.empty((len(indices), 3, height, width), dtype=np.float32)
    output_index = 0
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index != indices[output_index]:
                continue
            if frame.time is None or abs(float(frame.time) - expected_times[output_index]) > 1e-6:
                raise MetadataError("video_timeline_changed", str(path))
            if (frame.width, frame.height) != (spatial["source_width"], spatial["source_height"]):
                raise MetadataError("video_resolution_changed", str(path))
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            image = image.resize((spatial["resized_width"], spatial["resized_height"]), Image.Resampling.BILINEAR)
            left, top = spatial["crop_left"], spatial["crop_top"]
            rgb = np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32)
            video[output_index] = (rgb.transpose(2, 0, 1) / 127.5) - 1.
            output_index += 1
            if output_index == len(indices):
                break
    if output_index != len(indices):
        raise MetadataError("video_decode_short", f"Wanted {len(indices)}, read {output_index}")
    return video


def build_window_index(manifest, data_root, output, settings, *, seed=0, limit=0, shot_annotations=None):
    validate_settings(settings)
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    manifest, output, root = Path(manifest).resolve(), Path(output).resolve(), Path(data_root).resolve()
    annotations = {} if shot_annotations is None else shot_annotations
    if not isinstance(annotations, dict):
        raise ValueError("Shot annotations must be a JSON object keyed by video_path")
    annotations = {key.replace("\\", "/"): value for key, value in annotations.items()}
    original_manifest_hash = sha256_file(manifest)
    source_file = manifest.parent / "metadata_sources.json"
    original_sources_hash = sha256_file(source_file)
    dataset = RealCamMetadataDataset(manifest, root)
    (output / "timelines").mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    count = min(len(dataset), limit) if limit else len(dataset)
    for index in range(count):
        raw = dataset.records[index]
        window_diagnostics = None
        try:
            sample = dataset[index]
            path = Path(sample["video_path"])
            video_hash = sha256_file(path)
            timeline = scan_video(path, settings)
            times = timeline["timestamps"]
            for key in ("width", "height"):
                if source_value := sample.get(key):
                    if int(source_value) != timeline[key]:
                        raise MetadataError("video_metadata_mismatch", f"{key}: manifest={source_value}, video={timeline[key]}")
            if len(times) != len(sample["camera_extrinsics"]):
                raise MetadataError("camera_video_length_mismatch", f"video={len(times)}, camera={len(sample['camera_extrinsics'])}")
            camera_times = sample["timestamps"]
            if camera_times is not None and np.max(np.abs((times - times[0]) - (camera_times - camera_times[0]))) > settings["camera_time_tolerance_seconds"]:
                raise MetadataError("camera_video_timestamp_mismatch", raw["video_path"])
            annotated = annotations.get(raw["video_path"])
            if settings["shot_mode"] == "annotated" and annotated is None:
                raise MetadataError("missing_shot_annotation", raw["video_path"])
            cuts = sorted(set(timeline["detected_cuts"]) | set(validate_cuts([] if annotated is None else annotated, len(times))))
            starts, segments = feasible_starts(times, sample["valid_mask"], cuts, settings)
            if not len(starts):
                window_diagnostics = explain_window_rejection(times, sample["valid_mask"], cuts, settings, segments)
                raise MetadataError("no_valid_window", ", ".join(window_diagnostics["causes"]))
            if sha256_file(path) != video_hash:
                raise MetadataError("video_changed_during_scan", raw["video_path"])
            filename = f"timelines/{raw['sample_id']}.npz"
            np.savez_compressed(output / filename, timestamps=times, starts=starts, cuts=np.asarray(cuts, dtype=np.int64),
                                segments=np.asarray(segments, dtype=np.int64), cut_scores=timeline["cut_scores"])
            accepted.append({"metadata_index": index, "sample_id": raw["sample_id"], "video_path": raw["video_path"],
                             "video_sha256": video_hash, "timeline": filename, "timeline_sha256": sha256_file(output / filename),
                             "width": timeline["width"], "height": timeline["height"], "source_fps": timeline["fps"],
                             "num_frames": len(times), "feasible_windows": len(starts), "cut_count": len(cuts),
                             "shot_boundary_source": "annotation_and_detector" if annotated is not None and settings["shot_mode"] == "detect"
                             else "annotation" if settings["shot_mode"] == "annotated" else "detector"})
        except (OSError, ValueError, EOFError) as exc:
            rejected.append({"metadata_index": index, "sample_id": raw["sample_id"], "video_path": raw["video_path"],
                             "reason": exc.code if isinstance(exc, MetadataError) else type(exc).__name__, "detail": str(exc)})
            if window_diagnostics is not None:
                rejected[-1]["window_diagnostics"] = window_diagnostics
        if (index + 1) % 25 == 0:
            print(f"Window scan: {index + 1}/{count}; usable={len(accepted)}", flush=True)
    if sha256_file(manifest) != original_manifest_hash or sha256_file(source_file) != original_sources_hash:
        raise ValueError("Input manifest or metadata_sources.json changed during window preparation")
    index_data = {"schema_version": 1, "settings": settings, "seed": seed,
                  "manifest": manifest.relative_to(REPO_ROOT).as_posix() if manifest.is_relative_to(REPO_ROOT) else str(manifest),
                  "manifest_path_base": "repo" if manifest.is_relative_to(REPO_ROOT) else "absolute",
                  "manifest_sha256": original_manifest_hash, "metadata_sources_sha256": original_sources_hash,
                  "records": accepted, "input_samples": len(dataset), "inspected_samples": count,
                  "unused_annotation_keys": sorted(set(annotations) - {r["video_path"] for r in dataset.records}),
                  "rejected": rejected, "shot_detection_limitation": "Only provided/detected boundaries are excluded; automatic detection may miss cuts."}
    (output / "window_index.json").write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_data


class RealCamWindowDataset:
    """One random feasible window per clip, deterministic for seed/epoch/draw.

    NumPy video is float32 [F,C,H,W] in [-1,1], image equals video[0]. This
    indexable CPU dataset can be wrapped by the later Torch training loader.
    """
    def __init__(self, window_index, data_root, *, repo_root=REPO_ROOT, manifest_override=None):
        self.index_path = Path(window_index).resolve()
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index["schema_version"] != 1:
            raise ValueError("Unsupported window index schema")
        self.settings = self.index["settings"]
        validate_settings(self.settings)
        manifest = resolve_path(manifest_override or self.index["manifest"], repo_root)
        if sha256_file(manifest) != self.index["manifest_sha256"] or sha256_file(manifest.parent / "metadata_sources.json") != self.index["metadata_sources_sha256"]:
            raise ValueError("Source manifest changed; rebuild window index")
        self.metadata = RealCamMetadataDataset(manifest, data_root)
        self.epoch = 0
        self.verified_files = {}

    def __len__(self):
        return len(self.index["records"])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _verify(self, path, digest):
        stat = (path.stat().st_size, path.stat().st_mtime_ns)
        key = str(path)
        if self.verified_files.get(key) != (stat, digest):
            if sha256_file(path) != digest:
                raise ValueError(f"File changed since window preparation: {path}")
            self.verified_files[key] = (stat, digest)
        return stat

    def __getitem__(self, index):
        return self.get_sample(index, epoch=self.epoch)

    def get_sample(self, index, *, epoch=0, draw=0):
        record = self.index["records"][index]
        source = self.metadata[record["metadata_index"]]
        path = Path(source["video_path"])
        before_stat = self._verify(path, record["video_sha256"])
        timeline_path = data_path(record["timeline"], self.index_path.parent)
        self._verify(timeline_path, record["timeline_sha256"])
        with np.load(timeline_path, allow_pickle=False) as timeline:
            times, starts, segments, cuts = (timeline[name] for name in ("timestamps", "starts", "segments", "cuts"))
        digest = hashlib.sha256(f"{self.index['seed']}\0{record['sample_id']}\0{epoch}\0{draw}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        start = int(starts[rng.integers(len(starts))])
        end = next(int(end) for begin, end in segments if begin <= start < end)
        selected = window_indices(times, start, self.settings, end=end)
        if selected is None:
            raise ValueError("Cached start is no longer feasible; rebuild window index")
        indices, targets = selected
        spatial = spatial_transform(record["width"], record["height"], self.settings, rng)
        video = decode_window(path, indices, times[indices], spatial)
        if (path.stat().st_size, path.stat().st_mtime_ns) != before_stat:
            raise ValueError("Video changed during window decoding")
        camera_times = source["timestamps"]
        return {"video": video, "image": video[0].copy(), "prompt": source["caption"],
                "camera_intrinsics": transform_intrinsics(source["camera_intrinsics"][indices], source["intrinsics_units"], spatial),
                "camera_extrinsics": source["camera_extrinsics"][indices].copy(), "align_factor": source["align_factor"],
                "valid_mask": source["valid_mask"][indices].copy(), "frame_indices": indices,
                "timestamps": times[indices], "target_timestamps": targets, "relative_timestamps": times[indices] - times[start],
                "camera_timestamps": camera_times[indices] if camera_times is not None else times[indices],
                "metadata": {"sample_id": record["sample_id"], "video_path": record["video_path"],
                             "video_sha256": record["video_sha256"], "camera_sha256": source["camera_sha256"],
                             "source_id": source["source_id"], "group_id": source["group_id"], "group_basis": source["group_basis"],
                             "split": source["split"], "original_split": source["original_split"], "epoch": epoch, "draw": draw,
                             "window_start_frame": start, "target_fps": self.settings["target_fps"],
                             "intrinsics_units": "pixels", "camera_convention": source["camera_convention"],
                             "camera_coordinates_transformed": False, "align_factor_applied": False,
                             "camera_timestamp_source": "metadata" if camera_times is not None else "video_frame_association",
                             "spatial_transform": spatial, "shot_boundaries": cuts.tolist(),
                             "shot_boundary_source": record["shot_boundary_source"],
                             "max_time_error_seconds": float(np.max(np.abs(times[indices] - targets)))}}
