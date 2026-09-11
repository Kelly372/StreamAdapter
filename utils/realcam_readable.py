"""Human-readable snapshots of official NPZ metadata, without filtering rows."""
from collections import Counter, defaultdict
import json
import math

import numpy as np

from utils.realcam_dataset import load_official_camera_index, sha256_file
from utils.run_record import write_json


def readable_value(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "shape": list(value.shape),
                "values": readable_value(value.tolist())}
    if isinstance(value, np.generic):
        return readable_value(value.item())
    if isinstance(value, dict):
        return {str(key): readable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [readable_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported readable metadata value: {type(value).__name__}")


def export_readable_npz(source, folder, data_root, *, preview_count=3):
    """Export every record, keeping original order, paths, labels and array values."""
    before = sha256_file(source)
    records = load_official_camera_index(source, data_root)
    folder.mkdir(parents=True, exist_ok=False)
    source_counts, prefix_counts = Counter(), Counter()
    fields = defaultdict(lambda: {"present": 0, "types": Counter(), "array_shapes": Counter(), "array_dtypes": Counter()})
    previews = []
    with (folder / "records.jsonl").open("x", encoding="utf-8") as full, \
            (folder / "index.jsonl").open("x", encoding="utf-8") as brief:
        for index, item in enumerate(records.values()):
            label = str(item.get("dataset_source", "<missing>"))
            path = item["video_path"]
            source_counts[label] += 1
            # Descriptive only: this is not the archive's train/test membership.
            prefix_counts["/".join(path.replace("\\", "/").split("/")[:2])] += 1
            arrays = {}
            for key, value in item.items():
                stats = fields[key]
                stats["present"] += 1
                stats["types"][type(value).__name__] += 1
                if isinstance(value, np.ndarray):
                    stats["array_shapes"][str(list(value.shape))] += 1
                    stats["array_dtypes"][str(value.dtype)] += 1
                    arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            record = {"index": index, "metadata": readable_value(item)}
            full.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            short = {"index": index, "dataset_source": label, "video_path": path, "arrays": arrays}
            brief.write(json.dumps(short, ensure_ascii=False, allow_nan=False) + "\n")
            if index < preview_count:
                previews.append(record)
                print(json.dumps(short, ensure_ascii=False), flush=True)
    if before != sha256_file(source):
        raise ValueError(f"Source changed during readable export: {source}")
    write_json(folder / "preview.json", previews)
    summary = {"source": str(source), "source_sha256": before, "entries": len(records),
               "dataset_source_counts": dict(source_counts), "path_prefix_counts": dict(prefix_counts),
               "fields": {key: {name: dict(value) if isinstance(value, Counter) else value for name, value in stats.items()}
                          for key, stats in fields.items()},
               "preview_count": len(previews), "all_records_exported": True, "filtered": False,
               "notes": ["index is the zero-based original NPZ record position; source labels and paths are not rewritten",
                         "records.jsonl includes every field and all array values; index.jsonl omits array values",
                         "ndarrays store dtype/shape/values; tuples become lists; nonfinite floats use explicit tagged objects",
                         "path train/test components do not define the archive split; no videos or CSVs are read"]}
    write_json(folder / "summary.json", summary)
    (folder / "summary.txt").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Entries: {len(records)}; dataset_source: {dict(source_counts)}", flush=True)
    print(f"Top path prefixes (not archive splits): {dict(prefix_counts.most_common(10))}; full counts in summary.json", flush=True)
    print(f"Readable files: {folder}", flush=True)
    return summary
