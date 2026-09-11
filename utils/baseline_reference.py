"""Copy the exact stage-4 window beside an I2V result; never use it as conditioning."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def copy_reference(image, output, base_name, *, simple_names=False):
    """Resolve only the exporter's explicit sample mapping, including old exports."""
    image, output = Path(image).resolve(), Path(output).resolve()
    if Path(base_name).name != base_name or "\\" in base_name:
        raise ValueError("Reference name must be a single filename stem")
    record = {"input_image": image.as_posix(), "result_stem": base_name,
              "status": "gt_unavailable", "files": {}, "sha256": {},
              "note": "GT is the stage-4 cropped/resampled window, an H.264 preview; it is not a model input."}
    sources = {"input_image": (image, f"input{image.suffix.lower()}" if simple_names else f"{base_name}_input{image.suffix.lower()}")}
    # Stage 4 exports a flat baseline_i2v directory alongside samples/<stem>/.
    sample_dir = image.parent.parent / "samples" / image.stem
    metadata_path = sample_dir / "sample.json"
    if image.parent.name == "baseline_i2v" and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = f"baseline_i2v/{image.name}"
        if metadata.get("baseline_image", "").replace("\\", "/") != expected:
            raise ValueError(f"GT sample mapping does not match input image: {metadata_path}")
        if digest(sample_dir / "first_frame.png") != digest(image):
            raise ValueError(f"Input image differs from the exported GT first frame: {image}")
        sources.update({"ground_truth": (sample_dir / "gt.mp4", "ground_truth.mp4" if simple_names else f"{base_name}_gt.mp4"),
                        "sample_metadata": (metadata_path, "sample.json" if simple_names else f"{base_name}_sample.json")})
        record.update(status="gt_copied", source_sample=metadata_path.as_posix())
    # Verify all sources and destination conflicts before copying anything.
    for key, (source, name) in sources.items():
        checksum = digest(source)
        destination = output / name
        if destination.exists() and digest(destination) != checksum:
            raise FileExistsError(f"Refusing to replace a different reference: {destination}")
        record["files"][key] = name
        record["sha256"][key] = checksum
    for source, name in sources.values():
        if not (output / name).exists():
            shutil.copy2(source, output / name)
    (output / ("reference.json" if simple_names else f"{base_name}_reference.json")).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record
