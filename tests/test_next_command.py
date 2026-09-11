"""Verify successful CLI transitions and suppress suggestions on failure."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inspect_realcam
import prepare_realcam_windows as prepare
import run_baseline
import test_realcam_dataset
import test_realcam_windows
from utils.next_command import next_run_name
from utils.project_paths import REPO_ROOT


class NextCommandTests(unittest.TestCase):
    def test_inspection_success_prints_actual_manifest_and_failure_prints_none(self):
        fixture = test_realcam_dataset.RealCamTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.inputs([fixture.row("good")])
        repo = Path(fixture.temp.name) / "repo"
        args = ["--config", str(REPO_ROOT / "configs/data/realcam_inspection.yaml"),
                "--workspace-root", str(fixture.root.parents[1]), "--probe-mode", "metadata",
                "--set", "paths.data_root=public_data/RealCam-Vid", "--set", "data.validation_fraction=0", "--set", "data.group_by=source_id",
                "--run-name", "realestate10k_audit_smoke_v2"]
        console = io.StringIO()
        with patch.object(inspect_realcam, "REPO_ROOT", repo), redirect_stdout(console):
            self.assertEqual(inspect_realcam.main(args), 0)
        folder = repo / "output" / args[-1]
        record = json.loads((folder / "next_command.json").read_text())
        expected = ("python prepare_realcam_windows.py --manifest output/realestate10k_audit_smoke_v2/train.csv "
                    "--limit 50 --export-count 10 --run-name realestate10k_windows_smoke_v2")
        self.assertTrue(record["command"].startswith(expected))
        self.assertIn(record["command"], console.getvalue())
        self.assertIn("--workspace-root", record["argv"])
        self.assertEqual((folder / "next_command.txt").read_text().strip(), record["command"])
        (fixture.root / "RealEstate10K/good.mp4").unlink()
        args[-1] = "failed-audit"
        console = io.StringIO()
        with patch.object(inspect_realcam, "REPO_ROOT", repo), redirect_stdout(console):
            self.assertEqual(inspect_realcam.main(args), 1)
        self.assertNotIn("Next command", console.getvalue())
        self.assertFalse((repo / "output/failed-audit/next_command.json").exists())

    def test_index_only_suggests_export_before_baseline(self):
        fixture = test_realcam_windows.WindowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        relative = fixture.make_video()
        manifest = fixture.make_manifest([("source", relative)])
        repo = fixture.root / "repo"
        args = ["--config", str(REPO_ROOT / "configs/data/realcam_windows.yaml"), "--manifest", str(manifest),
                "--workspace-root", str(fixture.root / "workspace"), "--set", "paths.data_root=public_data/RealCam-Vid", "--set", "windows.height=16",
                "--set", "windows.width=24", "--export-count", "0", "--run-name", "windows_index"]
        with patch.object(prepare, "REPO_ROOT", repo), redirect_stdout(io.StringIO()):
            self.assertEqual(prepare.main(args), 0)
            first = json.loads((repo / "output/windows_index/next_command.json").read_text())["argv"]
            self.assertEqual(first[1], "prepare_realcam_windows.py")
            # Execute the suggested argv, as if copied at the documented repository cwd.
            # Absolute fixture inputs are unchanged; workspace suggestion is repository-relative.
            from utils.project_paths import resolve_path
            position = first.index("--workspace-root") + 1
            first[position] = str(resolve_path(first[position], repo))
            self.assertEqual(prepare.main(first[2:]), 0)
        name = first[first.index("--run-name") + 1]
        self.assertEqual(first[first.index("--export-count") + 1], "10")
        summary = json.loads((repo / "output" / name / "summary.json").read_text())
        self.assertEqual(summary["requested_exports"], 10)
        self.assertEqual(summary["exported_samples"], 1)
        self.assertEqual(summary["export_shortfall"], 9)
        second = json.loads((repo / "output" / name / "next_command.json").read_text())["argv"]
        self.assertEqual(second[1], "run_baseline.py")
        self.assertIn("--check-only", second)
        self.assertIn(f"data.data_path=output/{name}/baseline_i2v", second)
        self.assertTrue(list((repo / "output" / name / "baseline_i2v").glob("*.png")))

    def test_baseline_check_preserves_overrides_and_suppresses_failed_check(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            args = ["run_baseline.py", "--config", str(REPO_ROOT / "configs/baseline/longlive_bf16_i2v.yaml"),
                    "--workspace-root", str(repo / "custom workspace"), "--set", "seed=42", "--set", "num_samples=2",
                    "--set", "data.data_path=output/a b/baseline_i2v", "--check-only", "--run-name", "i2v_check_v2"]
            with patch.object(run_baseline, "REPO_ROOT", repo), patch("sys.argv", args), \
                    patch.object(run_baseline, "check_inputs", return_value=[]), redirect_stdout(io.StringIO()):
                self.assertEqual(run_baseline.main(), 0)
            command = json.loads((repo / "output/i2v_check_v2/next_command.json").read_text())
            self.assertNotIn("--check-only", command["argv"])
            self.assertNotIn("--dry-run", command["argv"])
            self.assertEqual(command["argv"][-1], "i2v_baseline_v2")
            for value in ("seed=42", "num_samples=2", "data.data_path=output/a b/baseline_i2v", str(repo / "custom workspace")):
                self.assertIn(value, command["argv"])
            self.assertIn("'data.data_path=output/a b/baseline_i2v'", command["command"])
            args[-1] = "failed-check"
            console = io.StringIO()
            with patch.object(run_baseline, "REPO_ROOT", repo), patch("sys.argv", args), \
                    patch.object(run_baseline, "check_inputs", return_value=["missing checkpoint"]), redirect_stdout(console):
                self.assertEqual(run_baseline.main(), 1)
            self.assertNotIn("Next command", console.getvalue())

    def test_next_name_avoids_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "windows_v2").mkdir()
            self.assertEqual(next_run_name(root / "audit_v2", "audit", "windows"), "windows_v2_2")


if __name__ == "__main__":
    unittest.main()
