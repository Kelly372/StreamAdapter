"""Temporary diagnostic coverage; removable with diagnose_realcam_temp.py."""
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

import diagnose_realcam_temp as diagnostic
from inspect_realcam import load_inspection_config
from utils.project_paths import REPO_ROOT
from utils.realcam_dataset import sha256_file


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.data = self.workspace / "public_data/RealCam-Vid"
        self.data.mkdir(parents=True)
        self.repo = self.workspace / "project/StreamAdapter"
        self.repo.mkdir(parents=True)
        self.config = str(REPO_ROOT / "configs/data/realcam_inspection.yaml")
        self.inputs = {}
        for split in ("train", "test"):
            rows = [{"dataset_source": "RealEstate10K", "video_path": f"RealEstate10K/{split}/scene-{split}/clip-{i}.mp4",
                     "short_caption": "room", "long_caption": "a room", "align_factor": 1., "camera_scale": 1., "vtss_score": .5}
                    for i in range(3)]
            self.inputs[split] = rows
            with (self.data / f"RealEstate10K_{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                path = self.data / row["video_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"Not decoded by this path-only diagnostic")
            self.save_npz(self.data / f"RealEstate10K_{split}.npz", rows)

    @staticmethod
    def save_npz(path, rows):
        payload = [dict(row, camera_intrinsics=np.array([1., 1., .5, .5]),
                        camera_extrinsics=np.tile(np.eye(4), (4, 1, 1))) for row in rows]
        np.savez_compressed(path, arr_0=np.array(payload, dtype=object))

    def run_diagnostic(self, extra=(), include_config=True):
        args = (["--config", self.config] if include_config else []) + ["--workspace-root", str(self.workspace), "--set", "paths.data_root=public_data/RealCam-Vid", *extra]
        console = io.StringIO()
        with patch.object(diagnostic, "REPO_ROOT", self.repo), patch.object(diagnostic, "environment_record", return_value={}), \
                redirect_stdout(console), redirect_stderr(console):
            code = diagnostic.main(args)
        folder = sorted((self.repo / "output").glob("diagnose_realcam_temp_*"))[-1]
        return code, folder, json.loads((folder / "report.json").read_text()), console.getvalue()

    def test_matching_paths_repeatable_read_only_and_no_video_decode(self):
        before = {str(path): sha256_file(path) for path in self.data.rglob("*") if path.is_file()}
        code, first, report, console = self.run_diagnostic()
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["rows"]), 6)
        self.assertEqual(len(report["archives"]), 2)
        self.assertIn("no video decoding", console)
        for name in ("parameters.json", "resolved_config.yaml", "diagnosis.txt", "report.json", "status.json"):
            self.assertTrue((first / name).is_file())
        code, second, _, _ = self.run_diagnostic()
        self.assertEqual(code, 0)
        self.assertNotEqual(first, second)
        self.assertEqual(before, {str(path): sha256_file(path) for path in self.data.rglob("*") if path.is_file()})

    def test_missing_selected_paths_reports_full_archive_and_prefix_difference(self):
        rows = self.inputs["test"]
        wrong = [dict(row, video_path=row["video_path"].replace("/test/", "/train/")) for row in rows]
        self.save_npz(self.data / "RealEstate10K_test.npz", wrong)
        self.save_npz(self.data / "RealCam-Vid_test.npz", rows)
        code, _, report, console = self.run_diagnostic(["--splits", "test"])
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertTrue(all(not row["exact_in_selected_npz"] for row in report["rows"]))
        full = next(item for item in report["archives"] if Path(item["path"]).name == "RealCam-Vid_test.npz")
        subset = next(item for item in report["archives"] if Path(item["path"]).name == "RealEstate10K_test.npz")
        self.assertEqual(full["rows"][0]["counts"]["exact"], 1)
        self.assertEqual(subset["rows"][0]["counts"]["same_scene_and_file"], 1)
        self.assertIn("selected_npz_exact=False", console)

    def test_previous_rejections_prioritized_and_actual_selected_archive_used(self):
        audit = self.repo / "output/old-audit"
        audit.mkdir(parents=True)
        config = load_inspection_config(self.config, str(self.workspace), ["paths.data_root=public_data/RealCam-Vid"])
        OmegaConf.save(config, audit / "resolved_config.yaml")
        selected = self.data / "previous-selected.npz"
        self.save_npz(selected, self.inputs["test"][:2])
        (audit / "metadata_sources.json").write_text(json.dumps({"RealEstate10K_test.csv": {
            "official_camera_npz": {"path": selected.name, "path_base": "data_root"}}}), encoding="utf-8")
        with (audit / "rejected.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["original_split", "video_path", "reason", "detail"])
            writer.writeheader()
            writer.writerow({"original_split": "test", "video_path": self.inputs["test"][2]["video_path"],
                             "reason": "missing_camera_metadata", "detail": "No NPZ entry"})
        code, folder, report, _ = self.run_diagnostic(["--audit-dir", "output/old-audit", "--splits", "test", "--limit", "1"],
                                                    include_config=False)
        self.assertEqual(code, 1)
        self.assertEqual(report["rows"][0]["canonical_video_path"], self.inputs["test"][2]["video_path"])
        self.assertEqual(report["rows"][0]["selected_npz"], str(selected))
        self.assertEqual(len(report["rows"]), 1)
        self.assertTrue((folder / "previous_rejected_examples.json").is_file())

    def test_missing_csv_failure_keeps_report(self):
        (self.data / "RealEstate10K_test.csv").unlink()
        code, folder, report, _ = self.run_diagnostic(["--splits", "test"])
        self.assertEqual(code, 2)
        self.assertTrue(report["errors"])
        self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
