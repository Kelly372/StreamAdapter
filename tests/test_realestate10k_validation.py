"""Exercise a fresh prepared dataset through the grouped CPU validation entry."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import av
import numpy as np
from omegaconf import OmegaConf

import validate_realestate10k as validation
from utils import project_paths
from utils.baseline_reference import digest
import test_realcam_partition


class ValidationTests(unittest.TestCase):
    def setUp(self):
        fixture = test_realcam_partition.PartitionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.repo = fixture.repo
        shutil.copytree(project_paths.REPO_ROOT / "configs", self.repo / "configs")
        # The dataset is ALREADY prepared. No extraction stage or earlier output is used.
        for split, rows in fixture.rows.items():
            np.savez_compressed(fixture.data / f"RealEstate10K_{split}.npz", arr_0=np.array(rows, dtype=object))
            for row in rows:
                path = fixture.data / row["video_path"]
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
        baseline_path = self.repo / "configs/baseline/longlive_bf16_i2v.yaml"
        config = OmegaConf.load(baseline_path)
        config.data.image_or_video_shape = [1, 32, 48, 2, 2]
        OmegaConf.save(config, baseline_path)
        # Keep this tiny fixture's one train group out of validation.
        config_path = self.repo / "configs/data/realcam_inspection.yaml"
        config = OmegaConf.load(config_path)
        config.data.validation_fraction = 0
        OmegaConf.save(config, config_path)
        self.stack = self.enterContext(ExitStack())
        for module in (validation, validation.inspect_realcam, validation.prepare_realcam_windows,
                       validation.run_baseline, project_paths):
            self.stack.enter_context(patch.object(module, "REPO_ROOT", self.repo))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))
        self.before = {str(p): digest(p) for p in fixture.data.rglob("*") if p.is_file()}

    def prepare(self, name="check", count=2):
        return validation.main(["--run-name", name, "--count", str(count), "--limit", "50",
                                "--workspace-root", str(self.fixture.workspace)])

    def test_clean_output_to_gt_and_one_inference_command(self):
        self.assertFalse((self.repo / "output").exists())
        self.assertEqual(self.prepare(), 0)
        folder = self.repo / "output/check"
        self.assertEqual(list((self.repo / "output").iterdir()), [folder])
        summary = validation.read_json(folder / "summary.json")
        self.assertEqual((summary["status"], summary["exported"], summary["split"]), ("prepared", 2, "test"))
        self.assertTrue((folder / "records/inspection/test.csv").is_file())
        self.assertFalse(list((folder / "records").glob("**/next_command.txt")))
        command = validation.read_json(folder / "records/next_command.json")["argv"]
        self.assertEqual(command, ["python", "validate_realestate10k.py", "--infer", "output/check"])
        for index in range(2):
            sample = folder / "comparisons" / f"{index:04d}"
            self.assertTrue((sample / "input.png").is_file())
            self.assertFalse((sample / "generated.mp4").exists())
            with av.open(str(sample / "ground_truth.mp4")) as container:
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 125)
        self.assertEqual(self.before, {str(p): digest(p) for p in self.fixture.data.rglob("*") if p.is_file()})
        # Verify dispatch and completion bookkeeping without pretending to run a GPU model.
        def fake_baseline(argv, *, output_dir, show_next):
            self.assertEqual(output_dir, folder / "records/inference")
            self.assertFalse(show_next)
            self.assertIn("inference.comparison_dir=output/check/comparisons", argv)
            for index in range(2):
                (folder / "comparisons" / f"{index:04d}" / "generated.mp4").write_bytes(b"mock GPU output")
            return 0
        with patch.object(validation.run_baseline, "main", side_effect=fake_baseline):
            self.assertEqual(validation.main(command[2:]), 0)
        self.assertEqual(validation.read_json(folder / "summary.json")["status"], "completed")

    def test_shortfall_has_no_inference_suggestion(self):
        self.assertEqual(self.prepare(count=3), 1)
        folder = self.repo / "output/check"
        summary = validation.read_json(folder / "summary.json")
        self.assertEqual(summary["status"], "preparation_failed")
        self.assertEqual(summary["exported"], 2)
        self.assertFalse((folder / "next_command.txt").exists())

    def test_missing_prepared_npz_never_uses_original_archive(self):
        (self.fixture.data / "RealEstate10K_test.npz").unlink()
        self.assertEqual(self.prepare(), 1)
        folder = self.repo / "output/check"
        self.assertIn("Prepared dataset file missing", validation.read_json(folder / "summary.json")["error"])
        self.assertFalse((folder / "records/inspection").exists())

    def test_changed_inputs_block_inference_before_model_load(self):
        self.assertEqual(self.prepare(), 0)
        folder = self.repo / "output/check"
        image = next((folder / "records/windows/baseline_i2v").glob("*.png"))
        image.write_bytes(b"changed input")
        with patch.object(validation.run_baseline, "main") as baseline:
            self.assertEqual(validation.main(["--infer", "output/check"]), 1)
            baseline.assert_not_called()
