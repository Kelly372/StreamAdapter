"""Build a reusable window index from a step-3 manifest and export I2V examples."""
import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import re
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from utils.project_paths import REPO_ROOT, add_path_arguments, resolve_path, workspace_root
from utils.realcam_dataset import sha256_file
from utils.realcam_windows import RealCamWindowDataset, build_window_index, validate_settings
from utils.run_record import environment_record, write_json


def load_window_config(path, workspace=None, overrides=()):
    config = OmegaConf.merge(OmegaConf.load(resolve_path(path, REPO_ROOT)), OmegaConf.from_dotlist(list(overrides)))
    root = workspace_root(workspace if workspace is not None else config.paths.workspace_root)
    config.paths.workspace_root = str(root)
    config.paths.repo_root = str(REPO_ROOT)
    config.paths.data_root = str(resolve_path(config.paths.data_root, root))
    OmegaConf.resolve(config)
    return config


def uint8_frame(frame):
    return np.clip(np.rint((frame.transpose(1, 2, 0) + 1.) * 127.5), 0, 255).astype(np.uint8)


def export_sample(sample, folder, baseline_folder, options):
    import av
    folder.mkdir(parents=True, exist_ok=False)
    name = folder.name
    Image.fromarray(uint8_frame(sample["image"])).save(folder / "first_frame.png")
    Image.fromarray(uint8_frame(sample["image"])).save(baseline_folder / f"{name}.png")
    # ImagePromptDataset interprets NEWLINES as shot changes; baseline text must
    # be a single prompt. Preserve the exact original text in sample.json.
    prompt = " ".join(sample["prompt"].split())
    if not prompt:
        raise ValueError("Cannot export an empty I2V prompt")
    (folder / "first_frame.txt").write_text(prompt, encoding="utf-8")
    (baseline_folder / f"{name}.txt").write_text(prompt, encoding="utf-8")
    fps = Fraction(str(sample["metadata"]["target_fps"]))
    video = sample["video"]
    with av.open(str(folder / "gt.mp4"), mode="w") as container:
        stream = container.add_stream(options["codec"], rate=fps)
        stream.width, stream.height, stream.pix_fmt = video.shape[3], video.shape[2], "yuv420p"
        stream.options = {"crf": str(options["crf"]), "preset": options["preset"]}
        for index, tensor in enumerate(video):
            frame = av.VideoFrame.from_ndarray(uint8_frame(tensor), format="rgb24")
            frame.pts, frame.time_base = index, 1 / fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    arrays = {key: sample[key] for key in ("camera_intrinsics", "camera_extrinsics", "align_factor", "valid_mask",
                                         "frame_indices", "timestamps", "relative_timestamps", "target_timestamps", "camera_timestamps")}
    np.savez_compressed(folder / "camera.npz", **arrays)
    metadata = dict(sample["metadata"], prompt=sample["prompt"], baseline_prompt=prompt,
                    frame_indices=sample["frame_indices"].tolist(), timestamps=sample["timestamps"].tolist(),
                    target_timestamps=sample["target_timestamps"].tolist(), video_shape=list(video.shape),
                    image_equals_first_video_frame=bool(np.array_equal(sample["image"], video[0])),
                    video_value_range="[-1,1] float32", gt_preview="H.264/YUV420 preview; PNG is the exact conditioning image",
                    baseline_image=f"baseline_i2v/{name}.png", baseline_prompt_file=f"baseline_i2v/{name}.txt")
    write_json(folder / "sample.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/realcam_windows.yaml")
    add_path_arguments(parser)
    parser.add_argument("--manifest", required=True, help="Step-3 train.csv, validation.csv or test.csv, relative to repository")
    parser.add_argument("--shot-annotations", help="Optional JSON: video_path -> first-frame indices of new shots; relative to repository")
    parser.add_argument("--limit", type=int, default=0, help="Inspect first N manifest clips, 0=all")
    parser.add_argument("--export-count", type=int, help="Number of usable clips to export; 0=index only")
    parser.add_argument("--run-name")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    overrides = list(args.overrides)
    if args.export_count is not None:
        overrides.append(f"export.count={args.export_count}")
    config = load_window_config(args.config, args.workspace_root, overrides)
    settings = OmegaConf.to_container(config.windows, resolve=True)
    validate_settings(settings)
    if args.limit < 0 or config.export.count < 0:
        raise ValueError("limit and export.count must be nonnegative")
    if config.export.count and (settings["height"] % 2 or settings["width"] % 2):
        raise ValueError("MP4 YUV420 preview export requires even width and height")
    manifest = resolve_path(args.manifest, REPO_ROOT)
    name = args.run_name or (f"windows_realestate10k_{manifest.stem}_f{settings['rgb_frames']}_fps{settings['target_fps']}_"
                             f"{settings['height']}x{settings['width']}_seed{config.seed}_{datetime.now():%Y%m%d-%H%M%S-%f}" +
                             (f"_{args.tag}" if args.tag else ""))
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", name) or name.endswith(".") or name.split(".")[0].upper() in reserved:
        raise ValueError("Run name must be a portable single folder name, max 180 characters")
    folder = REPO_ROOT / "output" / name
    folder.mkdir(parents=True, exist_ok=False)
    config.run = {"name": name, "output_folder": str(folder), "manifest": str(manifest), "limit": args.limit,
                  "shot_annotations": str(resolve_path(args.shot_annotations, REPO_ROOT)) if args.shot_annotations else None}
    OmegaConf.save(config, folder / "resolved_config.yaml", resolve=True)
    (folder / "source_config.yaml").write_text(resolve_path(args.config, REPO_ROOT).read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "sampling.txt").write_text("RealEstate10K window sampling (step 4)\n\n" + OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    write_json(folder / "launch.json", {"argv": sys.argv if argv is None else argv, "environment": environment_record()})
    write_json(folder / "status.json", {"status": "running"})
    print(f"Run directory: {folder}", flush=True)
    try:
        annotations = None
        if config.run.shot_annotations:
            path = Path(config.run.shot_annotations)
            annotations = json.loads(path.read_text(encoding="utf-8-sig"))
            write_json(folder / "shot_annotations.json", annotations)
            write_json(folder / "annotation_source.json", {"path": str(path), "sha256": sha256_file(path)})
        index = build_window_index(manifest, config.paths.data_root, folder, settings, seed=config.seed,
                                   limit=args.limit, shot_annotations=annotations)
        summary = {"input_samples": index["input_samples"], "inspected_samples": index["inspected_samples"],
                   "usable_clips": len(index["records"]), "feasible_windows": sum(row["feasible_windows"] for row in index["records"]),
                   "rejected_samples": len(index["rejected"]), "rejection_reasons": dict(Counter(row["reason"] for row in index["rejected"])),
                   "exported_samples": 0, "all_manifest_samples_inspected": index["input_samples"] == index["inspected_samples"],
                   "unused_annotation_keys": index["unused_annotation_keys"],
                   "shot_detection_limitation": index["shot_detection_limitation"]}
        write_json(folder / "summary.json", summary)
        write_json(folder / "rejected.json", index["rejected"])
        if not index["records"]:
            raise ValueError("No usable windows; see rejected.json and timing/shot settings")
        dataset = RealCamWindowDataset(folder / "window_index.json", config.paths.data_root)
        if config.export.count:
            (folder / "baseline_i2v").mkdir()
        for i in range(min(config.export.count, len(dataset))):
            sample = dataset.get_sample(i, epoch=config.export.epoch, draw=config.export.draw)
            sample_name = f"{i:04d}_{sample['metadata']['sample_id']}_s{sample['metadata']['window_start_frame']}"
            export_sample(sample, folder / "samples" / sample_name, folder / "baseline_i2v", OmegaConf.to_container(config.export))
            summary["exported_samples"] += 1
            write_json(folder / "summary.json", summary)
            del sample
        write_json(folder / "summary.json", summary)
        with (folder / "sampling.txt").open("a", encoding="utf-8") as handle:
            handle.write("\nResults:\n" + json.dumps(summary, indent=2))
        write_json(folder / "status.json", {"status": "completed_with_rejections" if index["rejected"] else "completed", "returncode": 0})
        print(f"Usable clips: {len(dataset)}; exported: {summary['exported_samples']}; rejected: {summary['rejected_samples']}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        write_json(folder / "status.json", {"status": "failed", "returncode": 2, "error": str(exc)})
        print(f"Window preparation failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
