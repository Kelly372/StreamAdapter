"""Temporal/geometry tests and real CPU video integration for step 4."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import inspect_realcam
import prepare_realcam_windows as prepare
from utils.realcam_dataset import probe_video
from utils.realcam_inspection import inspect_dataset
from utils.realcam_windows import (
    RealCamWindowDataset, build_window_index, decode_window, feasible_starts, scan_video,
    spatial_transform, transform_intrinsics, valid_segments, validate_settings, window_indices,
)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "workspace/public_data/RealCam-Vid"
        self.data.mkdir(parents=True)
        self.audit = self.root / "audit"
        self.audit.mkdir()
        self.output = self.root / "windows"
        self.output.mkdir()
        self.settings = OmegaConf.to_container(prepare.load_window_config("configs/data/realcam_windows.yaml").windows)
        self.settings.update(height=16, width=24, crop_mode="center")

    def make_video(self, name="clip", *, frames=160, fps=30, width=32, height=24, cut=None, timestamps=None):
        relative = f"RealEstate10K/train/{name}/clip.mkv"
        path = self.data / relative
        path.parent.mkdir(parents=True)
        # Lossless pixels allow direct verification of RGB / camera index pairing.
        with av.open(str(path), "w") as container:
            stream = container.add_stream("ffv1", rate=fps)
            stream.width, stream.height, stream.pix_fmt = width, height, "bgr0"
            if timestamps is not None:
                stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for i in range(frames):
                value = i % 240 if cut is None else (0 if i < cut else 255)
                array = np.full((height, width, 3), value, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                if timestamps is not None:
                    frame.pts, frame.time_base = round(timestamps[i] * 1000), Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return relative

    def make_manifest(self, videos, *, probe_mode="decode", invalid_pose=None, camera_fps=None):
        rows = []
        for name, relative in videos:
            observed = probe_video(self.data / relative)
            count = observed["num_frames"]
            extrinsics = np.tile(np.eye(4), (count, 1, 1))
            extrinsics[:, 0, 3] = np.arange(count)
            if invalid_pose is not None:
                extrinsics[invalid_pose, 0, 0] = -1
            camera_path = f"{name}.npz"
            extra = {} if camera_fps is None else {"timestamps": np.arange(count) / camera_fps}
            np.savez_compressed(self.data / camera_path, camera_intrinsics=np.array([.8, .9, .4, .6]),
                                camera_extrinsics=extrinsics, align_factor=2., **extra)
            rows.append({"dataset_source": "RealEstate10K", "source_video_id": name, "video_path": relative,
                         "long_caption": 'A room\nwith "windows".', "camera_path": camera_path})
        for split, split_rows in (("train", rows), ("test", [])):
            with (self.data / f"RealEstate10K_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(split_rows)
        config = inspect_realcam.load_inspection_config("configs/data/realcam_inspection.yaml",
            workspace=str(self.root / "workspace"), overrides=["data.validation_fraction=0", "inspection.progress_every=0",
                f"inspection.probe_mode={probe_mode}", "inspection.min_valid_pose_fraction=0.9"])
        inspect_dataset(config, self.audit)
        return self.audit / "train.csv"

    def build(self, manifest, annotations=None):
        return build_window_index(manifest, self.data, self.output, self.settings, seed=11, shot_annotations=annotations)

    def test_seven_column_csv_official_npz_to_relocated_window(self):
        headers = ["dataset_source", "video_path", "short_caption", "long_caption",
                   "align_factor", "camera_scale", "vtss_score"]
        caption = 'A room, with "windows".\nCamera moves forward.'
        for split in ("train", "test"):
            relative = f"RealEstate10K/{split}/scene-{split}/clip.mp4"
            path = self.data / relative
            path.parent.mkdir(parents=True)
            # Encode a real MP4 losslessly so each pixel carries its source frame index.
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264rgb", rate=24)
                stream.width, stream.height, stream.pix_fmt = 32, 24, "rgb24"
                stream.options = {"crf": "0", "preset": "ultrafast"}
                for i in range(125):
                    frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), i, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with (self.data / f"RealEstate10K_{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                # Exercise Windows CSV separators against POSIX NPZ keys.
                writer.writerow(["RealEstate10K", relative.replace("/", "\\"), "room", caption,
                                 3.67878591096826, 1.5096761946889783, .06628306])
            poses = np.tile(np.eye(4), (125, 1, 1))
            poses[:, 0, 3] = np.arange(125)
            payload = {"dataset_source": "RealEstate10K", "video_path": relative,
                       "camera_intrinsics": np.array([.8, .9, .4, .6]), "camera_extrinsics": poses,
                       "long_caption": "NPZ text must not replace the CSV text", "align_factor": 99.}
            np.savez_compressed(self.data / f"RealCam-Vid_{split}.npz", arr_0=np.array([payload], dtype=object))
        config = inspect_realcam.load_inspection_config("configs/data/realcam_inspection.yaml",
            workspace=str(self.root / "workspace"), overrides=["data.validation_fraction=0", "inspection.progress_every=0"])
        summary = inspect_dataset(config, self.audit)
        self.assertEqual(summary["accepted"], {"train": 1, "validation": 0, "test": 1})
        sources = json.loads((self.audit / "metadata_sources.json").read_text())
        self.assertEqual(sources["RealEstate10K_train.csv"]["columns"]["subset"], "dataset_source")
        # Relocate both data and inspected caches before loading the video window.
        relocated_data, relocated_audit = self.root / "moved/data", self.root / "moved/audit"
        shutil.copytree(self.data, relocated_data)
        shutil.copytree(self.audit, relocated_audit)
        build_window_index(relocated_audit / "train.csv", relocated_data, self.output, self.settings, seed=11)
        sample = RealCamWindowDataset(self.output / "window_index.json", relocated_data)[0]
        self.assertEqual(sample["prompt"], caption)
        self.assertEqual(sample["align_factor"], 3.67878591096826)
        self.assertEqual(sample["video"].shape, (125, 3, 16, 24))
        np.testing.assert_array_equal(sample["image"], sample["video"][0])
        np.testing.assert_array_equal(sample["camera_extrinsics"][:, 0, 3], sample["frame_indices"])
        np.testing.assert_array_equal(np.rint((sample["video"][:, 0, 0, 0] + 1) * 127.5), sample["frame_indices"])

    def test_24_and_30_fps_timing_and_no_low_fps_duplication(self):
        for fps, count in ((24, 125), (30, 156)):
            times = np.arange(count) / fps
            starts, _ = feasible_starts(times, np.ones(count, bool), [], self.settings)
            self.assertIn(0, starts)
            indices, targets = window_indices(times, 0, self.settings)
            self.assertEqual(len(indices), 125)
            self.assertTrue((np.diff(indices) > 0).all())
            self.assertLessEqual(np.abs(times[indices] - targets).max(), self.settings["max_time_error_seconds"])
        times = np.arange(200) / 20
        starts, _ = feasible_starts(times, np.ones(200, bool), [], self.settings)
        self.assertEqual(len(starts), 0)
        starts, _ = feasible_starts(np.arange(124) / 24, np.ones(124, bool), [], self.settings)
        self.assertEqual(len(starts), 0)

    def test_timestamp_jitter_gaps_invalid_masks_and_cuts(self):
        times = np.arange(300) / 30 + np.sin(np.arange(300)) * .001
        mask = np.ones(300, bool)
        mask[158] = False
        starts, segments = feasible_starts(times, mask, [200], self.settings)
        self.assertTrue(len(starts))
        for start in starts:
            end = next(end for begin, end in segments if begin <= start < end)
            indices, _ = window_indices(times, start, self.settings, end=end)
            self.assertFalse(indices[0] < 158 < indices[-1])
            self.assertFalse(indices[0] < 200 <= indices[-1])
            self.assertTrue(mask[indices].all())
        times[150:] += 1
        self.assertIn((0, 150), valid_segments(times, np.ones(300, bool), [], .0625))
        tiny = dict(self.settings, rgb_frames=2, target_fps=24)
        # Targets must not cross a cut even if clamping to the prior shot is within tolerance.
        self.assertIsNone(window_indices(np.arange(5) / 60, 0, tiny, end=2))

    def test_geometry_matches_half_pixel_resize_and_crop_projection(self):
        settings = dict(self.settings, height=20, width=20, crop_mode="center")
        spatial = spatial_transform(80, 40, settings, np.random.default_rng(1))
        self.assertEqual((spatial["resized_width"], spatial["resized_height"], spatial["crop_left"]), (40, 20, 10))
        source = np.array([[.8, .9, .4, .6]])
        transformed = transform_intrinsics(source, "normalized", spatial)[0]
        original = np.array([[64., 0, 32.], [0, 36., 24.], [0, 0, 1.]])
        point = np.array([.2, -.1, 2.])
        pixel = original @ point
        pixel /= pixel[2]
        expected = np.array([(pixel[0] + .5) * .5 - .5 - 10, (pixel[1] + .5) * .5 - .5])
        projected = transformed @ point
        np.testing.assert_allclose(projected[:2] / projected[2], expected)
        np.testing.assert_allclose(transform_intrinsics(original[None], "pixels", spatial)[0], transformed)
        self.assertEqual(source.tolist(), [[.8, .9, .4, .6]])

    def test_random_crop_is_reproducible_and_stretch_scales_axes(self):
        settings = dict(self.settings, crop_mode="random", width=20, height=20)
        a = spatial_transform(91, 37, settings, np.random.default_rng(12))
        b = spatial_transform(91, 37, settings, np.random.default_rng(12))
        self.assertEqual(a, b)
        stretch = spatial_transform(80, 40, dict(settings, resize_mode="stretch"), np.random.default_rng(1))
        self.assertEqual((stretch["crop_left"], stretch["crop_top"]), (0, 0))
        self.assertNotEqual(stretch["pixel_transform"][0][0], stretch["pixel_transform"][1][1])
        with self.assertRaises(ValueError):
            validate_settings(dict(settings, target_fps=0))

    def test_defaults_match_experiment_a_rgb_and_latent_dimensions(self):
        from utils.project_paths import load_config
        config = prepare.load_window_config("configs/data/realcam_windows.yaml")
        experiment = load_config("configs/camera_adapter/experiment_a.yaml")
        self.assertEqual(config.windows.rgb_frames, experiment.rgb_frames)
        self.assertEqual(config.windows.target_fps, experiment.target_fps)
        self.assertEqual(config.windows.height, experiment.image_or_video_shape[3] * 16)
        self.assertEqual(config.windows.width, experiment.image_or_video_shape[4] * 16)

    def test_real_decode_camera_indices_first_frame_and_epoch_reproducibility(self):
        relative = self.make_video(frames=210)
        manifest = self.make_manifest([("source", relative)])
        result = self.build(manifest)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["cut_count"], 0)  # smooth brightness drift is not a cut
        dataset = RealCamWindowDataset(self.output / "window_index.json", self.data)
        a, b = dataset[0], dataset[0]
        self.assertEqual(a["video"].shape, (125, 3, 16, 24))
        self.assertEqual(a["video"].dtype, np.float32)
        np.testing.assert_array_equal(a["video"], b["video"])
        np.testing.assert_array_equal(a["image"], a["video"][0])
        np.testing.assert_allclose(a["camera_extrinsics"][:, 0, 3], a["frame_indices"])
        recovered = np.rint((a["video"][:, 0, 0, 0] + 1) * 127.5)
        np.testing.assert_array_equal(recovered, a["frame_indices"])
        self.assertEqual(a["align_factor"], 2.)
        self.assertFalse(a["metadata"]["align_factor_applied"])
        self.assertTrue(a["valid_mask"].all())
        starts = {dataset.get_sample(0, epoch=epoch)["metadata"]["window_start_frame"] for epoch in range(5)}
        self.assertGreater(len(starts), 1)
        dataset.set_epoch(3)
        np.testing.assert_array_equal(dataset[0]["frame_indices"], dataset.get_sample(0, epoch=3)["frame_indices"])

    def test_pixel_marker_and_intrinsics_follow_same_spatial_transform(self):
        # A linear ramp makes Pillow's interior pixel-center mapping measurable.
        settings = dict(self.settings, width=16, height=16, resize_mode="stretch")
        spatial = spatial_transform(32, 24, settings, np.random.default_rng(1))
        ramp = np.tile(np.arange(32, dtype=np.uint8)[None, :, None] * 6, (24, 1, 3))
        resized = np.asarray(Image.fromarray(ramp).resize((16, 16), Image.Resampling.BILINEAR))
        matrix = np.asarray(spatial["pixel_transform"])
        for u in (3, 5, 9):
            original_u = (u - matrix[0, 2]) / matrix[0, 0]
            self.assertAlmostEqual(resized[8, u, 0] / 6, original_u, delta=1 / 6)

    def test_variable_frame_rate_uses_decoded_pts_not_average_fps(self):
        pts = np.concatenate([np.arange(80) / 60, 80 / 60 + np.arange(140) / 30])
        relative = self.make_video(frames=len(pts), timestamps=pts)
        manifest = self.make_manifest([("vfr", relative)])
        self.build(manifest)
        dataset = RealCamWindowDataset(self.output / "window_index.json", self.data)
        sample = dataset[0]
        observed = probe_video(self.data / relative)["timestamps"]
        np.testing.assert_allclose(sample["timestamps"], observed[sample["frame_indices"]])
        self.assertLessEqual(np.abs(sample["timestamps"] - sample["target_timestamps"]).max(), self.settings["max_time_error_seconds"])
        self.assertTrue((np.diff(sample["frame_indices"]) > 0).all())
        np.testing.assert_allclose(sample["camera_extrinsics"][:, 0, 3], sample["frame_indices"])
        self.assertGreater(np.max(np.abs(observed - np.arange(len(pts)) / 30)), .5)

    def test_invalid_pose_intervals_and_camera_time_mismatch_rejected(self):
        relative = self.make_video(frames=210)
        manifest = self.make_manifest([("invalid-pose", relative)], invalid_pose=100)
        result = self.build(manifest)
        self.assertEqual(result["rejected"][0]["reason"], "no_valid_window")
        manifest = self.make_manifest([("bad-time", relative)], probe_mode="metadata", camera_fps=25)
        result = self.build(manifest)
        self.assertEqual(result["rejected"][0]["reason"], "camera_video_timestamp_mismatch")

    def test_hard_cut_detection_and_explicit_annotations(self):
        relative = self.make_video(frames=250, fps=24, cut=125)
        timeline = scan_video(self.data / relative, self.settings)
        self.assertEqual(timeline["detected_cuts"], [125])
        manifest = self.make_manifest([("cut", relative)])
        result = self.build(manifest)
        with np.load(self.output / result["records"][0]["timeline"]) as cache:
            self.assertEqual(cache["starts"].tolist(), [0, 125])
        dataset = RealCamWindowDataset(self.output / "window_index.json", self.data)
        for epoch in range(3):
            sample = dataset.get_sample(0, epoch=epoch)
            self.assertEqual(len(np.unique(sample["video"])), 1)
        self.settings["shot_mode"] = "annotated"
        result = self.build(manifest)
        self.assertEqual(result["rejected"][0]["reason"], "missing_shot_annotation")
        result = self.build(manifest, {relative: [125]})
        self.assertEqual(result["records"][0]["shot_boundary_source"], "annotation")
        result = self.build(manifest, {relative: 0})
        self.assertEqual(result["rejected"][0]["reason"], "invalid_shot_boundaries")

    def test_no_window_short_video_and_corrupt_video_are_reported(self):
        relative = self.make_video(frames=60, fps=24)
        manifest = self.make_manifest([("short", relative)])
        result = self.build(manifest)
        self.assertFalse(result["records"])
        self.assertEqual(result["rejected"][0]["reason"], "no_valid_window")
        (self.data / relative).write_bytes(b"broken video")
        result = self.build(manifest)
        self.assertFalse(result["records"])
        self.assertEqual(len(result["rejected"]), 1)

    def test_portable_index_and_stale_video_cache_manifest_detection(self):
        relative = self.make_video(frames=160)
        manifest = self.make_manifest([("relocate", relative)])
        self.build(manifest)
        relocated = self.root / "relocated"
        shutil.copytree(self.data, relocated / "data")
        shutil.copytree(self.audit, relocated / "audit")
        shutil.copytree(self.output, relocated / "windows")
        dataset = RealCamWindowDataset(relocated / "windows/window_index.json", relocated / "data",
                                      manifest_override=relocated / "audit/train.csv")
        self.assertEqual(dataset[0]["video"].shape[0], 125)
        video = relocated / "data" / relative
        with video.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "File changed"):
            dataset[0]
        cache = self.output / dataset.index["records"][0]["timeline"]
        with cache.open("ab") as handle:
            handle.write(b"changed")
        original = RealCamWindowDataset(self.output / "window_index.json", self.data)
        with self.assertRaisesRegex(ValueError, "File changed"):
            original[0]
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Source manifest changed"):
            RealCamWindowDataset(self.output / "window_index.json", self.data)

    def test_cli_export_baseline_prompt_params_and_gt_frame_count(self):
        relative = self.make_video(frames=160)
        manifest = self.make_manifest([("cli", relative)])
        repo = self.root / "repo"
        config_path = prepare.REPO_ROOT / "configs/data/realcam_windows.yaml"
        args = ["--config", str(config_path), "--manifest", str(manifest), "--workspace-root", str(self.root / "workspace"),
                "--set", "windows.height=16", "--set", "windows.width=24", "--export-count", "1", "--run-name", "smoke"]
        with patch.object(prepare, "REPO_ROOT", repo):
            self.assertEqual(prepare.main(args), 0)
            with self.assertRaises(FileExistsError):
                prepare.main(args)
        folder = repo / "output/smoke"
        for filename in ("sampling.txt", "resolved_config.yaml", "source_config.yaml", "launch.json", "status.json", "summary.json", "window_index.json", "rejected.json"):
            self.assertTrue((folder / filename).is_file(), filename)
        sample_dir = next((folder / "samples").iterdir())
        metadata = json.loads((sample_dir / "sample.json").read_text())
        self.assertTrue(metadata["image_equals_first_video_frame"])
        self.assertIn("\n", metadata["prompt"])
        prompt = next((folder / "baseline_i2v").glob("*.txt")).read_text()
        self.assertNotIn("\n", prompt)
        self.assertEqual(probe_video(sample_dir / "gt.mp4")["num_frames"], 125)
        with np.load(sample_dir / "camera.npz") as camera:
            np.testing.assert_array_equal(camera["frame_indices"], metadata["frame_indices"])
        first = np.asarray(Image.open(sample_dir / "first_frame.png"))
        baseline = np.asarray(Image.open(next((folder / "baseline_i2v").glob("*.png"))))
        np.testing.assert_array_equal(first, baseline)

    def test_cli_no_usable_windows_failure_and_limit_are_recorded(self):
        short = self.make_video(name="short", frames=60)
        long = self.make_video(name="long", frames=160)
        manifest = self.make_manifest([("short", short), ("long", long)])
        repo = self.root / "repo"
        config_path = prepare.REPO_ROOT / "configs/data/realcam_windows.yaml"
        args = ["--config", str(config_path), "--manifest", str(manifest), "--workspace-root", str(self.root / "workspace"),
                "--limit", "1", "--export-count", "0", "--run-name", "no-window"]
        with patch.object(prepare, "REPO_ROOT", repo):
            self.assertEqual(prepare.main(args), 2)
        folder = repo / "output/no-window"
        status = json.loads((folder / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        summary = json.loads((folder / "summary.json").read_text())
        self.assertFalse(summary["all_manifest_samples_inspected"])
        self.assertEqual(summary["usable_clips"], 0)


if __name__ == "__main__":
    unittest.main()
