"""CSV metadata access for RealCam-Vid, without Torch or video preprocessing.

Camera arrays retain their original coordinate/scale convention. Step 4 will
sample synchronized windows; step 5 will transform cameras and build rays.
"""
import ast
import csv
import hashlib
import json
import math
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

from utils.project_paths import resolve_path


ALIASES = {
    "video_path": ("video_path", "video"),
    "caption": ("long_caption", "caption"),
    "short_caption": ("short_caption",),
    # Both confirmed CSVs and official NPZ entries use dataset_source.
    "subset": ("dataset_source", "data_source", "subset", "dataset"),
    "source_id": ("source_video_id", "source_id", "original_video_id", "youtube_id"),
    "clip_id": ("clip_id", "video_id"),
    "split": ("split",),
    "camera_path": ("camera_path", "camera_file"),
    "intrinsics": ("camera_intrinsics", "intrinsics", "K"),
    "extrinsics": ("camera_extrinsics", "extrinsics", "w2c"),
    "align_factor": ("align_factor",),
    "camera_scale": ("camera_scale",),
    "vtss_score": ("vtss_score",),
    "valid_mask": ("valid_mask", "camera_valid_mask"),
    "timestamps": ("timestamps", "frame_timestamps"),
    "fps": ("fps", "video_fps"),
    "num_frames": ("num_frames", "frame_count", "video_frames"),
    "width": ("width", "video_width"),
    "height": ("height", "video_height"),
}


class MetadataError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(detail)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_path(value, root):
    if not value or "://" in str(value):
        raise MetadataError("invalid_path", f"Expected a local dataset path, got {value!r}")
    path = resolve_path(value, root)
    if not path.is_relative_to(Path(root).resolve()):
        raise MetadataError("path_outside_data_root", str(value))
    return path


def relative_path(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def discover_csv(root, split, explicit=None, *, subset=None):
    if explicit:
        path = resolve_path(explicit, root)
        if not path.is_file():
            raise FileNotFoundError(f"{split} CSV does not exist: {path}")
        return data_path(path, root)
    root = Path(root)
    names = set()
    if subset:
        names = {f"{subset}_{split}".lower()}
        if subset.lower() == "realestate10k":
            names.add(f"realstate10k_{split}")  # filename spelling reported by the user
    # The confirmed layout places subset CSVs directly under RealCam-Vid.
    # Do not let nested backup/export CSVs shadow these files or cause ambiguity.
    matches = sorted(path for path in root.glob("*.csv") if path.stem.lower() in names)
    if not matches:
        matches = sorted(path for path in root.rglob("*.csv")
                         if split in re.split(r"[^a-z0-9]+", path.stem.lower()))
        dedicated = [path for path in matches if path.stem.lower() in names]
        if dedicated:
            matches = dedicated
    if len(matches) != 1:
        raise ValueError(f"Expected one {split} CSV under {root}, found {len(matches)}: "
                         f"{[str(p) for p in matches]}. Set data.{split}_csv explicitly.")
    return matches[0]


def resolve_columns(headers, overrides=None, *, external_camera=False):
    if len(set(headers)) != len(headers) or any(not header for header in headers):
        raise ValueError("CSV header contains duplicate or empty column names")
    overrides = overrides or {}
    unknown = set(overrides) - set(ALIASES)
    if unknown:
        raise ValueError(f"Unknown canonical column mappings: {sorted(unknown)}")
    columns = {}
    for key, aliases in ALIASES.items():
        name = overrides[key] if key in overrides else next((v for v in aliases if v in headers), None)
        if name is not None and name not in headers:
            raise ValueError(f"Mapped column {key}={name!r} is missing. Available: {headers}")
        columns[key] = name
    if not columns["video_path"]:
        raise ValueError("CSV needs video_path (or data.columns.video_path mapping)")
    if not external_camera and not columns["camera_path"] and not (columns["intrinsics"] and columns["extrinsics"]):
        raise ValueError("CSV has no camera arrays. Configure data.train_camera_npz / test_camera_npz to join official NPZ by video_path, or provide camera_path / inline camera columns.")
    return columns


def field(row, columns, name):
    column = columns.get(name)
    return row.get(column, "").strip() if column else ""


def csv_rows(path, *, encoding="utf-8-sig", delimiter=",", field_limit=16777216):
    """Yield logical rows and seekable text offsets, including quoted newlines."""
    csv.field_size_limit(int(field_limit))
    with Path(path).open(encoding=encoding, newline="") as handle:
        # readline iteration keeps tell() available, unlike TextIO.__next__.
        reader = csv.reader(iter(handle.readline, ""), delimiter=delimiter, strict=True)
        headers = next(reader, None)
        if not headers:
            raise ValueError(f"CSV is empty: {path}")
        headers = [item.strip() for item in headers]
        yield headers
        row_number = 1
        while True:
            offset = handle.tell()
            values = next(reader, None)
            if values is None:
                break
            if not values:
                continue
            row_number += 1
            yield offset, row_number, values


def parse_array(value, name):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                # pandas may serialize NumPy arrays with spaces instead of commas.
                tokens = re.findall(r"\[|\]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
                if not tokens or "".join(tokens) != re.sub(r"[\s,]+", "", value):
                    raise MetadataError("invalid_array", f"{name}: unsupported or truncated array text") from exc
                joined = []
                for token in tokens:
                    if joined and joined[-1] != "[" and token != "]":
                        joined.append(",")
                    joined.append(token)
                try:
                    value = ast.literal_eval("".join(joined))
                except (ValueError, SyntaxError) as parse_exc:
                    raise MetadataError("invalid_array", f"{name}: invalid numeric list") from parse_exc
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MetadataError("invalid_array", f"Invalid numeric {name}") from exc
    if array.size == 0:
        raise MetadataError("empty_array", name)
    return array


def _numeric_array(value, root, name):
    if isinstance(value, str) and value.lower().endswith(".npy"):
        path = data_path(value, root)
        if not path.is_file():
            raise MetadataError("missing_camera_file", str(path))
        value = np.load(path, allow_pickle=False)
    return parse_array(value, name)


def load_camera(row, columns, root, *, rotation_tolerance=0.01, timestamps_unit="seconds", external_payload=None):
    payload = external_payload if external_payload is not None else {}
    camera_file = field(row, columns, "camera_path")
    if camera_file:
        path = data_path(camera_file, root)
        if not path.is_file():
            raise MetadataError("missing_camera_file", str(path))
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                wanted = {name for key in ("intrinsics", "extrinsics", "align_factor", "valid_mask", "timestamps")
                          for name in ALIASES[key]}
                payload = {key: archive[key] for key in archive.files if key in wanted}
        else:
            raise MetadataError("camera_file_format", "camera_path supports JSON or numeric NPZ (no pickle)")
        if not isinstance(payload, dict):
            raise MetadataError("camera_file_format", "Camera JSON must be an object")

    def value(key, default=None):
        inline = field(row, columns, key)
        if inline:
            return inline
        return next((payload[name] for name in ALIASES[key] if name in payload), default)

    extrinsics = _numeric_array(value("extrinsics"), root, "extrinsics")
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise MetadataError("extrinsics_shape", f"Expected [F,4,4], got {extrinsics.shape}")
    frames = len(extrinsics)
    intrinsics = _numeric_array(value("intrinsics"), root, "intrinsics")
    if intrinsics.shape == (4,) or intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (frames,) + intrinsics.shape).copy()
    if intrinsics.shape not in ((frames, 4), (frames, 3, 3)):
        raise MetadataError("intrinsics_shape", f"Expected [4], [F,4], [3,3] or [F,3,3], got {intrinsics.shape}")
    try:
        align_factor = float(value("align_factor"))
    except (TypeError, ValueError):
        raise MetadataError("invalid_align_factor", "align_factor must be an explicit positive scalar") from None
    if not math.isfinite(align_factor) or align_factor <= 0:
        raise MetadataError("invalid_align_factor", str(align_factor))

    valid = np.isfinite(extrinsics).all(axis=(1, 2))
    valid &= np.isclose(extrinsics[:, 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0).all(axis=1)
    rotation = np.where(np.isfinite(extrinsics[:, :3, :3]), extrinsics[:, :3, :3], 0)
    valid &= np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2)) <= rotation_tolerance
    valid &= np.abs(np.linalg.det(rotation) - 1) <= rotation_tolerance
    valid &= np.isfinite(intrinsics).reshape(frames, -1).all(axis=1)
    if intrinsics.ndim == 2:
        valid &= (intrinsics[:, :2] > 0).all(axis=1)
    else:
        valid &= (intrinsics[:, 0, 0] > 0) & (intrinsics[:, 1, 1] > 0)
        valid &= np.isclose(intrinsics[:, 2, :], [0, 0, 1], atol=1e-5, rtol=0).all(axis=1)
    mask = value("valid_mask")
    if mask is not None:
        mask = parse_array(mask, "valid_mask")
        if mask.shape != (frames,) or not np.isin(mask, [0, 1]).all():
            raise MetadataError("invalid_mask", "valid_mask must contain one boolean/0/1 per frame")
        valid &= mask.astype(bool)
    timestamps = value("timestamps")
    if timestamps is not None:
        timestamps = parse_array(timestamps, "timestamps")
        unit_scale = {"seconds": 1.0, "milliseconds": 0.001, "microseconds": 0.000001}[timestamps_unit]
        timestamps = timestamps * unit_scale
        if timestamps.shape != (frames,) or not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
            raise MetadataError("invalid_timestamps", "Expected finite, strictly increasing per-frame timestamps")
    return {"camera_intrinsics": intrinsics, "camera_extrinsics": extrinsics,
            "align_factor": align_factor, "valid_mask": valid, "timestamps": timestamps}


def numeric_field(row, columns, name, integer=False):
    raw = field(row, columns, name)
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        raise MetadataError("invalid_video_metadata", f"{name}={raw!r}") from None
    if not math.isfinite(number) or number <= 0 or (integer and number != int(number)):
        raise MetadataError("invalid_video_metadata", f"{name}={raw!r}")
    return int(number) if integer else number


def probe_video(path, mode="decode"):
    """Decode without retaining pixels; header mode explicitly offers weaker QA."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Video inspection requires PyAV: python -m pip install av==13.1.0") from exc
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise MetadataError("no_video_stream", str(path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        width, height = stream.width, stream.height
        timestamps = None
        if mode == "header" and stream.frames:
            frames = stream.frames
        else:
            times = []
            for frame in container.decode(stream):
                if (frame.width, frame.height) != (width, height):
                    raise MetadataError("variable_resolution", str(path))
                times.append(float(frame.time) if frame.time is not None else math.nan)
            frames = len(times)
            timestamps = np.asarray(times, dtype=np.float64)
            if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
                raise MetadataError("invalid_video_timestamps", str(path))
        if not frames or not width or not height or fps is None or not math.isfinite(fps) or fps <= 0:
            raise MetadataError("invalid_video_metadata", str(path))
    return {"num_frames": int(frames), "fps": fps, "width": width, "height": height,
            "timestamps": timestamps, "verification": "decode" if timestamps is not None else "header"}


def contiguous_runs(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def source_partition(source_id, seed, validation_fraction):
    # Stable under row reorder and dataset growth; no test samples enter val.
    digest = hashlib.sha256(f"{seed}\0{source_id}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if score < validation_fraction else "train"


def camera_digest(camera):
    digest = hashlib.sha256()
    for name in ("camera_intrinsics", "camera_extrinsics", "valid_mask", "timestamps", "align_factor"):
        value = camera[name]
        if value is None:
            digest.update(b"none")
        else:
            value = np.asarray(value, dtype="<f8")
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
    return digest.hexdigest()


class _NumpyMetadataUnpickler(pickle.Unpickler):
    """The official arr_0 uses NumPy objects; reject all other pickle globals."""
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            from numpy.core import multiarray
            return getattr(multiarray, name)
        raise ValueError(f"Unsupported object in camera metadata NPZ: {module}.{name}")


def load_official_camera_index(path, root):
    """Load one official split archive, matched by rooted video path, never row order.

    Official object archives must be materialized in RAM once per split. They are
    not repeatedly opened by the downstream dataset; accepted arrays are cached.
    """
    with zipfile.ZipFile(path) as archive:
        with archive.open("arr_0.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported official metadata NPY version: {version}")
            if not dtype.hasobject or len(shape) != 1:
                raise ValueError("Official metadata must be a 1D arr_0 array of dictionaries")
            records = _NumpyMetadataUnpickler(handle).load()
    if not isinstance(records, np.ndarray) or records.shape != shape:
        raise ValueError("Metadata array does not match its header")
    index = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("video_path"), str):
            raise ValueError("Every official metadata entry requires video_path")
        key = relative_path(data_path(item["video_path"], root), root)
        if key in index:
            raise ValueError(f"Duplicate video_path in camera metadata NPZ: {key}")
        index[key] = item
    return index


class RealCamMetadataDataset:
    """Read an inspected manifest; lazily retrieve full camera annotations.

    This is an indexable metadata dataset, not yet a video tensor/window loader.
    CSV content hashes prevent stale offsets from silently selecting another row.
    """
    def __init__(self, manifest_path, data_root):
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path)
        self.sources = json.loads((self.manifest_path.parent / "metadata_sources.json").read_text(encoding="utf-8"))
        with self.manifest_path.open(encoding="utf-8", newline="") as handle:
            self.records = list(csv.DictReader(handle))
        used = {record["metadata_csv"] for record in self.records}
        self.source_stats = {}
        for name in used:
            path = data_path(name, self.data_root)
            if sha256_file(path) != self.sources[name]["sha256"]:
                raise ValueError(f"Metadata CSV changed since inspection: {name}. Rebuild the manifests.")
            self.source_stats[name] = (path.stat().st_size, path.stat().st_mtime_ns)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        source = self.sources[record["metadata_csv"]]
        csv.field_size_limit(source["field_size_limit"])
        path = data_path(record["metadata_csv"], self.data_root)
        if (path.stat().st_size, path.stat().st_mtime_ns) != self.source_stats[record["metadata_csv"]]:
            raise ValueError(f"Metadata CSV changed while reading: {path}")
        with path.open(encoding=source["encoding"], newline="") as handle:
            handle.seek(int(record["metadata_offset"]))
            values = next(csv.reader(iter(handle.readline, ""), delimiter=source["delimiter"], strict=True))
        row = dict(zip(source["headers"], values))
        if len(values) != len(source["headers"]) or relative_path(
                data_path(field(row, source["columns"], "video_path"), self.data_root), self.data_root) != record["video_path"]:
            raise ValueError("Manifest offset does not match its video; rebuild the manifests")
        if record.get("camera_cache"):
            # Cache timestamps are already converted to seconds by inspection.
            camera = load_camera({"camera_path": record["camera_cache"]}, {"camera_path": "camera_path"},
                                 self.manifest_path.parent.resolve(), rotation_tolerance=source["rotation_tolerance"])
        else:
            camera = load_camera(row, source["columns"], self.data_root,
                                 rotation_tolerance=source["rotation_tolerance"], timestamps_unit=source["timestamps_unit"])
        if camera_digest(camera) != record["camera_sha256"]:
            raise ValueError(f"Camera annotations changed since inspection: {record['video_path']}")
        record.update(camera)
        record["video_path"] = str(data_path(record["video_path"], self.data_root))
        record["camera_convention"] = source["camera_convention"]
        record["intrinsics_units"] = source["intrinsics_units"]
        return record
