import json
from pathlib import Path
import tempfile
import unittest


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

    def test_grouped_names_are_readable(self):
        result = copy_reference(self.image, self.run, "generated", simple_names=True)
        self.assertEqual(result["files"]["ground_truth"], "ground_truth.mp4")
        self.assertTrue((self.run / "input.png").is_file())
        self.assertTrue((self.run / "sample.json").is_file())
        self.assertTrue((self.run / "reference.json").is_file())
