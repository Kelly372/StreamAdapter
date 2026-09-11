"""Step 3A: preserve metadata exactly, then feed subset NPZs into inspection."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import extract_realestate10k as extraction
import inspect_realcam
from utils.realcam_dataset import discover_csv, load_official_camera_index, sha256_file


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "public_data/RealCam-Vid"
        self.data.mkdir(parents=True)

    def entries(self, split):
        # The test archive deliberately points into an original train directory.
        return [{"dataset_source": subset, "video_path": f"{subset}/train/group-{split}/clip.mp4",
                 "long_caption": "a room", "short_caption": "room", "align_factor": 1.25,
                 "camera_intrinsics": np.array([1, 1, .5, .5], dtype=np.float32),
                 "camera_extrinsics": np.tile(np.eye(4, dtype=np.float64), (125, 1, 1)),
                 "camera_scale": 0., "vtss_score": .9, "extra": {"preserve": [1, "x", float("nan")]}}
                for subset in ("DL3DV", "RealEstate10K", "MiraData")]

    def make_input(self, split):
        path = self.data / f"RealCam-Vid_{split}.npz"
        np.savez_compressed(path, arr_0=np.array(self.entries(split), dtype=object))
        return path

    def test_roundtrip_preserves_all_fields_order_dtype_and_source_file(self):
        source = self.make_input("test")
        entries = self.entries("test")
        second = self.entries("test")[1]
        second["video_path"] = second["video_path"].replace("clip.mp4", "clip-2.mp4")
        second["camera_extrinsics"][:, 0, 3] = 7
        entries.append(second)
        np.savez_compressed(source, np.array(entries, dtype=object))
        original_hash = sha256_file(source)
        destination = self.root / "RealEstate10K_test.npz"
        report = extraction.extract_split(source, destination, self.data)
        records = list(load_official_camera_index(destination, self.data).values())
        self.assertEqual(report["selected_entries"], 2)
        self.assertEqual(report["excluded_entries"], 2)
        self.assertTrue(extraction.metadata_equal(self.entries("test")[1], records[0]))
        self.assertTrue(extraction.metadata_equal(second, records[1]))
        self.assertIn("/train/", records[0]["video_path"])
        self.assertEqual(sha256_file(source), original_hash)
        with self.assertRaises(FileExistsError):
            extraction.extract_split(source, destination, self.data)

    def test_missing_source_labels_and_empty_subset_fail(self):
        path = self.root / "bad.npz"
        row = self.entries("train")[1]
        del row["dataset_source"]
        np.savez(path, np.array([row], dtype=object))
        with self.assertRaisesRegex(ValueError, "Missing dataset_source"):
            extraction.extract_split(path, self.root / "missing.npz", self.data)
        np.savez(path, np.array([self.entries("train")[0]], dtype=object))
        with self.assertRaisesRegex(ValueError, "No RealEstate10K"):
            extraction.extract_split(path, self.root / "empty.npz", self.data)

    def test_inspect_only_defaults_to_test_and_exports_all_fields_without_filtering(self):
        source = self.make_input("test")
        before = sha256_file(source)
        console = io.StringIO()
        repo = self.root / "repo"
        with patch.object(extraction, "REPO_ROOT", repo), redirect_stdout(console):
            self.assertEqual(extraction.main(["--inspect-only", "--workspace-root", str(self.root),
                                             "--preview-count", "2", "--run-name", "readable"]), 0)
        folder = repo / "output/readable"
        records = [json.loads(line) for line in (folder / "test/records.jsonl").read_text().splitlines()]
        self.assertEqual([row["metadata"]["dataset_source"] for row in records], ["DL3DV", "RealEstate10K", "MiraData"])
        self.assertEqual([row["index"] for row in records], [0, 1, 2])
        for raw, restored in zip(self.entries("test"), records):
            restored = restored["metadata"]
            self.assertEqual(set(raw), set(restored))
            self.assertEqual(restored["video_path"], raw["video_path"])
            for key in ("camera_intrinsics", "camera_extrinsics"):
                self.assertEqual(restored[key]["dtype"], str(raw[key].dtype))
                self.assertEqual(restored[key]["shape"], list(raw[key].shape))
                np.testing.assert_array_equal(restored[key]["values"], raw[key])
            self.assertEqual(restored["extra"]["preserve"][-1], {"__nonfinite_float__": "NaN"})
        self.assertEqual(len(json.loads((folder / "test/preview.json").read_text())), 2)
        self.assertEqual(len((folder / "test/index.jsonl").read_text().splitlines()), 3)
        self.assertEqual(sha256_file(source), before)
        self.assertFalse((folder / "RealEstate10K_test.npz").exists())
        self.assertNotIn("Next: python inspect_realcam.py", console.getvalue())
        summary = json.loads((folder / "test/summary.json").read_text())
        self.assertEqual(summary["path_prefix_counts"]["RealEstate10K/train"], 1)
        self.assertTrue(summary["all_records_exported"])
        self.assertFalse(summary["filtered"])

    def test_csv_subset_preference_and_ambiguous_subset_files(self):
        for name in ("RealCam-Vid_test.csv", "RealState10K_test.csv"):
            (self.data / name).touch()
        self.assertEqual(discover_csv(self.data, "test", subset="RealEstate10K").name, "RealState10K_test.csv")
        self.assertEqual(discover_csv(self.data, "test", "RealCam-Vid_test.csv", subset="RealEstate10K").name, "RealCam-Vid_test.csv")
        (self.data / "RealEstate10K_test.csv").touch()
        with self.assertRaisesRegex(ValueError, "found 2"):
            discover_csv(self.data, "test", subset="RealEstate10K")

    def test_cli_both_splits_feed_inspector_without_moving_data(self):
        for split in ("train", "test"):
            self.make_input(split)
            row = self.entries(split)[1]
            video = self.data / row["video_path"]
            video.parent.mkdir(parents=True)
            video.touch()  # metadata mode intentionally does not decode
            csv_row = {key: row[key] for key in ("dataset_source", "video_path", "long_caption", "short_caption", "align_factor", "camera_scale", "vtss_score")}
            with (self.data / f"RealEstate10K_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=csv_row)
                writer.writeheader()
                writer.writerow(csv_row)
            # Extra full table must not cause ambiguity or be selected.
            (self.data / f"RealCam-Vid_{split}.csv").write_text("unused", encoding="utf-8")
        repo = self.root / "repo"
        with patch.object(extraction, "REPO_ROOT", repo):
            args = ["--workspace-root", str(self.root), "--run-name", "subset"]
            self.assertEqual(extraction.main(args), 0)
            with self.assertRaises(FileExistsError):
                extraction.main(args)
        folder = repo / "output/subset"
        for name in ("RealEstate10K_train.npz", "RealEstate10K_test.npz", "extraction.txt", "parameters.json", "launch.json", "summary.json", "status.json"):
            self.assertTrue((folder / name).is_file())
        config_path = inspect_realcam.REPO_ROOT / "configs/data/realcam_inspection.yaml"
        with patch.object(inspect_realcam, "REPO_ROOT", repo):
            self.assertEqual(inspect_realcam.main([
                "--config", str(config_path), "--workspace-root", str(self.root), "--camera-metadata-dir", "output/subset",
                "--set", "data.validation_fraction=0", "--probe-mode", "metadata", "--run-name", "audit"]), 0)
        sources = json.loads((repo / "output/audit/metadata_sources.json").read_text())
        self.assertEqual(Path(sources["RealEstate10K_test.csv"]["official_camera_npz"]["path"]), folder / "RealEstate10K_test.npz")
        summary = json.loads((repo / "output/audit/summary.json").read_text())
        self.assertEqual(summary["accepted"], {"train": 1, "validation": 0, "test": 1})

    def test_single_split_and_preflight_failure_are_recorded(self):
        source = self.make_input("test")
        with patch.object(extraction, "REPO_ROOT", self.root / "repo"):
            self.assertEqual(extraction.main(["--splits", "test", "--test-npz", str(source), "--run-name", "test-only"]), 0)
            self.assertEqual(extraction.main(["--workspace-root", str(self.root), "--run-name", "missing-train"]), 2)
        self.assertFalse((self.root / "repo/output/test-only/RealEstate10K_train.npz").exists())
        status = json.loads((self.root / "repo/output/missing-train/status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_splits"], [])


if __name__ == "__main__":
    unittest.main()
