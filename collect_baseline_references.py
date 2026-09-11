"""Add GT comparisons to an existing baseline run without rerunning the model."""
import argparse
from datetime import datetime
import re
import sys

from omegaconf import OmegaConf

from utils.baseline_reference import copy_reference
from utils.project_paths import REPO_ROOT, resolve_path
from utils.run_record import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing output directory, relative to the repository")
    parser.add_argument("--data-path", help="Original baseline_i2v directory after migration; relative to the repository")
    args = parser.parse_args(argv)
    folder = resolve_path(args.run_dir, REPO_ROOT)
    if not folder.is_dir() or not folder.is_relative_to((REPO_ROOT / "output").resolve()):
        parser.error("--run-dir must be an existing directory inside repository output/")
    audit = folder / f"reference_backfill_{datetime.now():%Y%m%d-%H%M%S-%f}"
    audit.mkdir()
    write_json(audit / "parameters.json", {"argv": argv if argv is not None else sys.argv[1:],
                                          "run_dir": str(folder), "data_path_override": args.data_path,
                                          "matching": "Original sorted ImagePromptDataset image list and saved prompt; inputs must be unchanged"})
    results, errors = [], []
    try:
        config = OmegaConf.load(folder / "resolved_config.yaml")
        if not config.get("i2v") or not config.get("save_with_index"):
            raise ValueError("Backfill requires an I2V run with save_with_index=true")
        data = resolve_path(args.data_path or config.get("data_path") or config.data.data_path, REPO_ROOT)
        write_json(audit / "effective_config.json", {"data_path": str(data),
                   "run_config": OmegaConf.to_container(config, resolve=True)})
        if (data / "video").is_dir():
            raise ValueError("Backfill supports ImagePromptDataset exports, not the video dataset layout")
        image_dir = data / "images" if (data / "images").is_dir() else data
        prompt_dir = data / "prompts" if (data / "prompts").is_dir() else image_dir
        extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        images = sorted({p for ext in extensions for pattern in (f"*{ext}", f"*{ext.upper()}")
                         for p in image_dir.glob(pattern)}, key=lambda p: p.name)
        outputs = []
        for path in sorted(folder.iterdir()):
            match = re.fullmatch(r"rank\d+-(\d+)-\d+_(regular|ema|lora)\.(mp4|pt)", path.name)
            if path.is_file() and match:
                outputs.append((path, int(match[1])))
        if not outputs:
            raise ValueError("No generated indexed videos or latents found")
        for path, index in outputs:
            try:
                image = images[index]
                source_prompt = (prompt_dir / f"{image.stem}.txt").read_text(encoding="utf-8").strip()
                saved_prompt = (folder / f"{path.stem}_prompts.txt").read_text(encoding="utf-8")
                saved_lines = [re.sub(r"^\[\d+(?:,\d+)*\] ", "", line)
                               for line in saved_prompt.splitlines() if line.strip()]
                # Stage-4 captions are a single line; multiple shots cannot be inferred safely here.
                if len(source_prompt.splitlines()) != 1 or saved_lines != [source_prompt]:
                    raise ValueError("Saved prompt does not match the original single-shot export")
                record = copy_reference(image, folder, path.stem)
                results.append(record)
                print(f"{path.name}: {record['files'].get('ground_truth', 'GT unavailable')}")
            except (OSError, ValueError, IndexError) as exc:
                errors.append({"result": path.name, "error": str(exc)})
        unavailable = sum(r["status"] != "gt_copied" for r in results)
        write_json(audit / "summary.json", {"results": results, "errors": errors, "gt_unavailable": unavailable})
        success = not errors and not unavailable
        write_json(audit / "status.json", {"status": "completed" if success else "incomplete"})
        print(f"GT copied: {len(results) - unavailable}; unavailable: {unavailable}; errors: {len(errors)}. Details: {audit}")
        return 0 if success else 1
    except (OSError, ValueError, KeyError) as exc:
        write_json(audit / "status.json", {"status": "failed", "error": str(exc)})
        print(f"Reference collection failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
