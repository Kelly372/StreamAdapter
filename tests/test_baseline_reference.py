import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

import collect_baseline_references as collect
from utils.baseline_reference import copy_reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        self.windows = self.repo / "output" / "windows"
        self.data = self.windows / "baseline_i2v"
        self.sample = self.windows / "samples" / "example"
        self.run = self.repo / "output" / "baseline"
        for path in (self.data, self.sample, self.run):
            path.mkdir(parents=True)
        self.image = self.data / "example.png"
        self.image.write_bytes(b"identical exported first frame")
        (self.sample / "first_frame.png").write_bytes(self.image.read_bytes())
        (self.sample / "gt.mp4").write_bytes(b"GT preview bytes, must be copied unchanged")
        (self.sample / "sample.json").write_text(json.dumps({"baseline_image": "baseline_i2v/example.png",
                                                           "frame_indices": [4, 5, 7]}))
        (self.data / "example.txt").write_text("A moving camera.")

    def test_copy_and_repeat_preserve_exact_gt_and_relative_links(self):
        result = copy_reference(self.image, self.run, "rank0-0-0_regular")
        self.assertEqual(result["status"], "gt_copied")
        for key, original in (("ground_truth", self.sample / "gt.mp4"), ("input_image", self.image)):
            self.assertEqual((self.run / result["files"][key]).read_bytes(), original.read_bytes())
        self.assertEqual(copy_reference(self.image, self.run, "rank0-0-0_regular"), result)
        (self.run / result["files"]["ground_truth"]).write_bytes(b"another GT")
        with self.assertRaises(FileExistsError):
            copy_reference(self.image, self.run, "rank0-0-0_regular")

    def test_modified_conditioning_image_cannot_get_wrong_gt(self):
        self.image.write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "differs"):
            copy_reference(self.image, self.run, "example")
        self.assertFalse(list(self.run.iterdir()))

    def test_plain_image_has_explicit_unavailable_status(self):
        image = self.repo / "plain.png"
        image.write_bytes(b"plain input")
        result = copy_reference(image, self.run, "example")
        self.assertEqual(result["status"], "gt_unavailable")
        self.assertNotIn("ground_truth", result["files"])

    def test_backfill_matches_saved_index_and_prompt_without_model(self):
        OmegaConf.save(OmegaConf.create({"i2v": True, "save_with_index": True,
                                        "data_path": "missing-old-workspace"}), self.run / "resolved_config.yaml")
        generated = self.run / "rank0-0-0_regular.mp4"
        generated.write_bytes(b"generated video, leave untouched")
        (self.run / "rank0-0-0_regular_prompts.txt").write_text("[0,1,2,3] A moving camera.\n")
        args = ["--run-dir", "output/baseline", "--data-path", "output/windows/baseline_i2v"]
        with patch.object(collect, "REPO_ROOT", self.repo):
            self.assertEqual(collect.main(args), 0)
            self.assertEqual(collect.main(args), 0)
        self.assertEqual(generated.read_bytes(), b"generated video, leave untouched")
        self.assertEqual((self.run / "rank0-0-0_regular_gt.mp4").read_bytes(), (self.sample / "gt.mp4").read_bytes())
        (self.data / "example.txt").write_text("Unrelated sample")
        with patch.object(collect, "REPO_ROOT", self.repo):
            self.assertEqual(collect.main(args), 1)
