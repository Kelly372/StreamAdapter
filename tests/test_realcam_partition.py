"""CSV-driven partitioning and new subset-root loading through actual video windows."""
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from omegaconf import OmegaConf

import extract_realestate10k as extraction
import inspect_realcam
from utils.project_paths import REPO_ROOT
from utils.realcam_dataset import load_official_camera_index, sha256_file
from utils.realcam_partition import subset_video_path
from utils.realcam_windows import RealCamWindowDataset, build_window_index


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.source = self.workspace / "public_data/RealCam-Vid"
        self.data = self.source / "RealEstate10K"
        self.data.mkdir(parents=True)
        self.repo = self.workspace / "project/StreamAdapter"
        self.repo.mkdir(parents=True)
        self.rows = {"train": [self.row("train/train-scene/a.mp4", 125)],
                     "test": [self.row("test/test-scene/b.mp4", 160), self.row("train/cross-scene/c.mp4", 130)]}
        for split in self.rows:
            self.write_csv(split, self.rows[split])
        # CSV test rows are in BOTH source archives; directory train/test also differs.
        self.write_npz("train", [self.rows["test"][0]])
        self.write_npz("test", [self.rows["test"][1], self.rows["train"][0]])

    def row(self, key, frames):
        ext = np.tile(np.eye(4), (frames, 1, 1))
        ext[:, 0, 3] = np.arange(frames)
        return {"dataset_source": "RealEstate10K", "video_path": key, "short_caption": "room", "long_caption": "A room",
                "align_factor": 2., "camera_scale": 1., "vtss_score": .5,
                "camera_intrinsics": np.array([.8, .9, .4, .6], np.float32), "camera_extrinsics": ext}

    def write_csv(self, split, rows):
        fields = ["dataset_source", "video_path", "short_caption", "long_caption", "align_factor", "camera_scale", "vtss_score"]
        with (self.data / f"RealEstate10K_{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def write_npz(self, split, rows):
        values = [dict(row, video_path="RealEstate10K/" + row["video_path"]) for row in rows]
        np.savez_compressed(self.source / f"RealCam-Vid_{split}.npz", arr_0=np.array(values, dtype=object))

    def run_partition(self, name):
        with patch.object(extraction, "REPO_ROOT", self.repo), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = extraction.main(["--workspace-root", str(self.workspace), "--run-name", name])
        return code, self.repo / "output" / name

    def test_csv_membership_order_arrays_and_relocated_stage3_stage4(self):
        before = {path: sha256_file(path) for path in self.source.rglob("*") if path.is_file()}
        code, folder = self.run_partition("partition")
        self.assertEqual(code, 0)
        for split, rows in self.rows.items():
            restored = load_official_camera_index(folder / f"RealEstate10K_{split}.npz", self.data)
            self.assertEqual(list(restored), [row["video_path"] for row in rows])
            for row in rows:
                self.assertTrue(extraction.metadata_equal(row, restored[row["video_path"]]))
            shutil.copy2(folder / f"RealEstate10K_{split}.npz", self.data)
        self.assertEqual(json.loads((folder / "test_provenance.json").read_text())[0]["source_archives"], ["train"])
        self.assertEqual(json.loads((folder / "test_provenance.json").read_text())[1]["source_archives"], ["test"])
        self.assertTrue(all(sha256_file(path) == digest for path, digest in before.items()))
        self.assertFalse((folder / "_partition_staging.sqlite3").exists())
        for rows in self.rows.values():
            for row in rows:
                path = self.data / row["video_path"]
                path.parent.mkdir(parents=True)
                with av.open(str(path), "w") as container:
                    stream = container.add_stream("libx264rgb", rate=24)
                    stream.width, stream.height, stream.pix_fmt = 32, 24, "rgb24"
                    stream.options = {"crf": "0", "preset": "ultrafast"}
                    for i in range(len(row["camera_extrinsics"])):
                        frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), i, np.uint8), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
        # Move the full subset layout before stage 3; no absolute paths in the inputs.
        moved_workspace = Path(self.temp.name) / "moved"
        moved_data = moved_workspace / "public_data/RealCam-Vid/RealEstate10K"
        shutil.copytree(self.data, moved_data)
        with patch.object(inspect_realcam, "REPO_ROOT", self.repo), redirect_stdout(io.StringIO()):
            self.assertEqual(inspect_realcam.main(["--config", str(REPO_ROOT / "configs/data/realcam_inspection.yaml"),
                "--workspace-root", str(moved_workspace), "--set", "data.validation_fraction=0", "--run-name", "audit"]), 0)
        audit = self.repo / "output/audit"
        summary = json.loads((audit / "summary.json").read_text())
        self.assertEqual(summary["accepted"], {"train": 1, "validation": 0, "test": 2})
        settings = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "configs/data/realcam_windows.yaml").windows)
        settings.update(height=24, width=32, crop_mode="center")
        windows = self.repo / "output/windows"
        windows.mkdir()
        build_window_index(audit / "test.csv", moved_data, windows, settings)
        dataset = RealCamWindowDataset(windows / "window_index.json", moved_data)
        self.assertEqual(len(dataset), 2)
        for i in range(len(dataset)):
            sample = dataset[i]
            self.assertEqual(sample["video"].shape, (125, 3, 24, 32))
            np.testing.assert_array_equal(sample["image"], sample["video"][0])
            np.testing.assert_array_equal(sample["camera_extrinsics"][:, 0, 3], sample["frame_indices"])
            np.testing.assert_array_equal(np.rint((sample["video"][:, 0, 0, 0] + 1) * 127.5), sample["frame_indices"])

    def test_missing_or_conflicting_metadata_never_publishes_archives(self):
        self.write_npz("train", [])
        code, folder = self.run_partition("missing")
        self.assertEqual(code, 2)
        self.assertFalse(list(folder.glob("*.npz")))
        self.assertEqual(json.loads((folder / "partition_report.json").read_text())["issues"][0]["reason"], "missing_camera_metadata")
        bad = dict(self.rows["train"][0], align_factor=99.)
        self.write_npz("train", [self.rows["test"][0], bad])
        code, folder = self.run_partition("conflict")
        self.assertEqual(code, 2)
        self.assertFalse(list(folder.glob("*.npz")))
        self.assertEqual(json.loads((folder / "partition_report.json").read_text())["issues"][0]["reason"], "conflicting_camera_metadata")
        self.write_npz("train", [self.rows["test"][0], self.rows["train"][0]])
        code, folder = self.run_partition("identical")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((folder / "partition_report.json").read_text())["sources"]["test"]["identical_duplicates"], 1)

    def test_duplicate_csv_paths_fail_before_partitioning(self):
        self.write_csv("test", [self.rows["train"][0]])
        code, folder = self.run_partition("duplicate-csv")
        self.assertEqual(code, 2)
        self.assertFalse(list(folder.glob("*.npz")))
        self.assertIn("Duplicate video_path", json.loads((folder / "partition_report.json").read_text())["issues"][0]["detail"])

    def test_normalization_is_only_prefix_and_separator_conversion(self):
        self.assertEqual(subset_video_path(r"RealEstate10K\test\scene\a.mp4", self.data, original=True), "test/scene/a.mp4")
        self.assertEqual(subset_video_path("./train/scene/a.mp4", self.data), "train/scene/a.mp4")
        self.assertEqual(subset_video_path(str(self.data / "test/scene/a.mp4"), self.data), "test/scene/a.mp4")
        for path in ("RealEstate10K/train/scene/a.mp4", "../train/scene/a.mp4", "test/a.mp4"):
            with self.assertRaises(ValueError):
                subset_video_path(path, self.data)


if __name__ == "__main__":
    unittest.main()
