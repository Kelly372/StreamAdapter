"""CPU coverage for camera payloads, split integrity, portable indexes and video QA."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import inspect_realcam
from utils.realcam_dataset import (
    MetadataError, RealCamMetadataDataset, contiguous_runs, data_path, discover_csv,
    load_camera, load_official_camera_index, parse_array, probe_video, resolve_columns,
)
from utils.realcam_inspection import inspect_dataset


CONFIG = "configs/data/realcam_inspection.yaml"


class RealCamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace" / "public_data" / "RealCam-Vid"
        self.root.mkdir(parents=True)
        self.output = Path(self.temp.name) / "report"
        self.output.mkdir()
        self.config = inspect_realcam.load_inspection_config(
            CONFIG, workspace=str(self.root.parents[1]), overrides=[
                "paths.data_root=public_data/RealCam-Vid", "data.train_csv=train.csv", "data.group_by=source_id", "inspection.probe_mode=metadata",
                "data.validation_fraction=0", "inspection.progress_every=0"])

    def row(self, name, source=None, frames=125, **changes):
        path = self.root / "RealEstate10K" / f"{name}.mp4"
        path.parent.mkdir(exist_ok=True)
        path.touch()
        row = {"video_path": path.relative_to(self.root).as_posix(),
               "source_video_id": source or name, "long_caption": '房间, with a "window"\nsecond line',
               "camera_intrinsics": "[1, 1, 0.5, 0.5]",
               "camera_extrinsics": json.dumps(np.broadcast_to(np.eye(4), (frames, 4, 4)).tolist()),
               "align_factor": "2.5", "fps": "24", "num_frames": str(frames), "width": "16", "height": "16"}
        row.update(changes)
        return row

    def inputs(self, train, test=None):
        if test is None:
            test = [self.row("held-out")]
        for split, rows in (("train", train), ("test", test)):
            headers = list(dict.fromkeys(key for row in rows for key in row))
            with (self.root / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

    def inspect(self, probe=probe_video):
        return inspect_dataset(self.config, self.output, probe=probe)

    @staticmethod
    def fake_probe(path, mode):
        return {"num_frames": 125, "fps": 24., "width": 16, "height": 16,
                "timestamps": np.arange(125) / 24, "verification": mode}

    def test_array_formats_and_unsafe_text(self):
        expected = np.array([[1., -0.25], [2e-3, 4]])
        for text in ('[[1, -.25], [2e-3, 4]]', '[[1. -0.25]\n [2e-3 4.]]', expected.tolist()):
            np.testing.assert_allclose(parse_array(text, "K"), expected)
        for text in ("__import__('os').getcwd()", "[[1 ... 2]]", "array([1, 2])", "[bad]"):
            with self.assertRaises(MetadataError):
                parse_array(text, "K")

    def test_camera_masks_shapes_scale_and_timestamps(self):
        row = self.row("a", frames=4, valid_mask="[1, 0, 1, 1]", timestamps="[0, 100, 200, 300]")
        columns = resolve_columns(row)
        camera = load_camera(row, columns, self.root, timestamps_unit="milliseconds")
        self.assertEqual(contiguous_runs(camera["valid_mask"]), [(0, 1), (2, 4)])
        np.testing.assert_allclose(camera["timestamps"], [0, .1, .2, .3])
        self.assertEqual(camera["align_factor"], 2.5)
        self.assertEqual(camera["camera_intrinsics"].shape, (4, 4))
        for changes in ({"align_factor": "nan"}, {"align_factor": "0"}, {"camera_extrinsics": "[[1, 2]]"},
                        {"camera_intrinsics": "[1,2,3]"}, {"valid_mask": "[1]"}, {"timestamps": "[0,0,1,2]"}):
            with self.assertRaises(MetadataError):
                load_camera(dict(row, **changes), columns, self.root)
        ext = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
        ext[2, 0, 0] = -1  # reflection is not a valid rotation
        camera = load_camera(dict(row, camera_extrinsics=json.dumps(ext.tolist())), columns, self.root)
        self.assertEqual(camera["valid_mask"].tolist(), [True, False, False, True])

    def test_numeric_camera_files_and_no_pickle(self):
        ext = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        np.savez(self.root / "camera.npz", camera_intrinsics=[1, 1, .5, .5], camera_extrinsics=ext,
                 align_factor=2, caption=np.array({"irrelevant": True}, dtype=object))
        row = {"video_path": "a.mp4", "camera_path": "camera.npz"}
        camera = load_camera(row, resolve_columns(row), self.root)
        np.testing.assert_array_equal(camera["camera_extrinsics"], ext)
        np.save(self.root / "poses.npy", ext)
        (self.root / "camera.json").write_text(json.dumps({"intrinsics": [1, 1, .5, .5],
            "extrinsics": "poses.npy", "align_factor": 3}), encoding="utf-8")
        row["camera_path"] = "camera.json"
        self.assertEqual(load_camera(row, resolve_columns(row), self.root)["align_factor"], 3)
        np.savez(self.root / "pickle.npz", camera_extrinsics=np.array({"unsafe": True}, dtype=object))
        row["camera_path"] = "pickle.npz"
        with self.assertRaises(ValueError):
            load_camera(row, resolve_columns(row), self.root)

    def test_portable_lazy_manifest_multiline_and_hashes(self):
        self.inputs([self.row("first"), self.row("second")])
        summary = self.inspect()
        self.assertEqual(summary["accepted"], {"train": 2, "validation": 0, "test": 1})
        self.assertEqual(summary["verification_levels"], {"metadata_only": 3})
        copied = Path(self.temp.name) / "relocated" / "RealCam-Vid"
        shutil.copytree(self.root, copied)
        dataset = RealCamMetadataDataset(self.output / "train.csv", copied)
        sample = dataset[1]
        self.assertEqual(sample["caption"], '房间, with a "window"\nsecond line')
        self.assertEqual(sample["source_id"], "second")
        self.assertEqual(sample["camera_extrinsics"].shape, (125, 4, 4))
        self.assertTrue(Path(sample["video_path"]).is_relative_to(copied))
        with (copied / "train.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "changed while reading"):
            dataset[0]
        with self.assertRaisesRegex(ValueError, "changed since inspection"):
            RealCamMetadataDataset(self.output / "train.csv", copied)

    def test_external_camera_change_rejected(self):
        row = self.row("external")
        payload = {key: json.loads(row.pop(key)) for key in ("camera_intrinsics", "camera_extrinsics", "align_factor")}
        path = self.root / "cam.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        row["camera_path"] = "cam.json"
        self.inputs([row])
        self.inspect()
        dataset = RealCamMetadataDataset(self.output / "train.csv", self.root)
        self.assertEqual(dataset[0]["align_factor"], 2.5)
        payload["align_factor"] = 5
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Camera annotations changed"):
            dataset[0]

    def test_source_groups_and_reorder_stability(self):
        self.config.data.validation_fraction = .5
        train = [self.row(f"source-{i}-clip-{clip}", source=f"source-{i}") for i in range(20) for clip in range(2)]
        self.inputs(train)
        self.inspect()
        partitions = {}
        for split in ("train", "validation", "test"):
            dataset = RealCamMetadataDataset(self.output / f"{split}.csv", self.root)
            partitions[split] = {row["source_id"] for row in dataset.records}
        self.assertTrue(partitions["train"] and partitions["validation"])
        self.assertFalse(partitions["train"] & partitions["validation"])
        self.assertEqual(partitions["test"], {"held-out"})
        self.inputs(list(reversed(train)))
        self.inspect()
        for split in ("train", "validation"):
            dataset = RealCamMetadataDataset(self.output / f"{split}.csv", self.root)
            self.assertEqual({row["source_id"] for row in dataset.records}, partitions[split])

    def test_train_test_overlap_detected_beyond_limit(self):
        self.config.inspection.limit_per_split = 1
        self.inputs([self.row("a", source="overlap"), self.row("uninspected")],
                    [self.row("b"), self.row("c", source="overlap")])
        summary = self.inspect()
        self.assertEqual(summary["overlapping_train_test_sources"], ["overlap"])
        self.assertEqual(summary["rejection_reasons"]["source_split_overlap"], 1)
        self.assertFalse(summary["all_selected_rows_inspected"])
        self.assertEqual(summary["accepted"]["train"], 0)

    def test_duplicate_paths_and_missing_source_are_rejected(self):
        a = self.row("a")
        b = dict(a, video_path=a["video_path"].replace("/", "\\"), source_video_id="different-id")
        self.inputs([a, b, self.row("no-source", source_video_id="")])
        summary = self.inspect()
        self.assertEqual(summary["rejection_reasons"], {"missing_source_id": 1, "duplicate_video": 2})

    def test_subset_filter_and_explicit_source_regex(self):
        row = self.row("original_clip-1", source_video_id="")
        self.config.data.source_id_regex = r"RealEstate10K/(?P<source_id>[^/]+)_clip-\d+\.mp4$"
        self.inputs([row, self.row("ignore", subset="DL3DV")], [self.row("test_clip-1", source_video_id="")])
        summary = self.inspect()
        self.assertEqual(summary["scan"]["train"]["other_subsets"], 1)
        self.assertEqual(RealCamMetadataDataset(self.output / "train.csv", self.root)[0]["source_id"], "original")

    def test_decode_metadata_and_camera_alignment_errors(self):
        self.config.inspection.probe_mode = "decode"
        self.inputs([self.row("length", frames=124, num_frames=""), self.row("fps", fps="30"),
                     self.row("timestamps", timestamps=json.dumps((np.arange(125) / 30).tolist())),
                     self.row("width", width="32"), self.row("good")])
        summary = self.inspect(probe=self.fake_probe)
        self.assertEqual(summary["accepted"]["train"], 1)
        self.assertEqual(summary["rejection_reasons"], {"camera_video_length_mismatch": 1, "fps_mismatch": 1,
                                                       "camera_video_timestamp_mismatch": 1, "video_metadata_mismatch": 1})

    def test_short_and_invalid_clips(self):
        missing = self.row("missing")
        (self.root / missing["video_path"]).unlink()
        self.inputs([self.row("short", frames=10), self.row("bad", camera_intrinsics="[-1,1,.5,.5]"), missing])
        summary = self.inspect()
        self.assertEqual(summary["duration_eligibility"], {"no": 1, "yes": 1})
        self.assertEqual(summary["rejection_reasons"], {"invalid_camera_fraction": 1, "missing_video": 1})

    def test_column_mapping_and_paths(self):
        row = self.row("mapped")
        row["media"] = row.pop("video_path")
        self.config.data.columns = {"video_path": "media"}
        self.inputs([row], [dict(self.row("test"), media="RealEstate10K/test.mp4")])
        self.assertEqual(self.inspect()["accepted"]["train"], 1)
        with self.assertRaises(MetadataError):
            data_path("../outside.mp4", self.root)
        (self.root / "extra_train.csv").touch()
        with self.assertRaisesRegex(ValueError, "found 2"):
            discover_csv(self.root, "train")
        config = inspect_realcam.load_inspection_config(CONFIG, workspace=str(self.root.parents[1]),
                                                       overrides=["paths.data_root=public_data/RealCam-Vid", "data.train_csv=train.csv"])
        self.assertEqual(Path(config.data.train_csv), self.root / "train.csv")

    def test_cli_outputs_and_failure_status_no_overwrite(self):
        self.inputs([self.row("good")])
        repo = Path(self.temp.name) / "fake_repo"
        args = ["--config", str(inspect_realcam.REPO_ROOT / CONFIG), "--workspace-root", str(self.root.parents[1]),
                "--probe-mode", "metadata", "--set", "paths.data_root=public_data/RealCam-Vid", "--set", "data.validation_fraction=0", "--set", "data.train_csv=train.csv",
                "--run-name", "audit"]
        with patch.object(inspect_realcam, "REPO_ROOT", repo):
            self.assertEqual(inspect_realcam.main(args), 0)
            with self.assertRaises(FileExistsError):
                inspect_realcam.main(args)
            (self.root / "test.csv").unlink()
            args[-1] = "missing-input"
            self.assertEqual(inspect_realcam.main(args), 2)
        folder = repo / "output" / "audit"
        for name in ("source_config.yaml", "resolved_config.yaml", "inspection.txt", "launch.json", "status.json",
                     "metadata_sources.json", "summary.json", "rejected.csv", "train.csv", "validation.csv", "test.csv"):
            self.assertTrue((folder / name).is_file(), name)
        self.assertEqual(json.loads((folder / "status.json").read_text())["probe_mode"], "metadata")
        failed = json.loads((repo / "output" / "missing-input" / "status.json").read_text())
        self.assertEqual(failed["status"], "failed")

    def test_official_npz_join_by_path_and_portable_camera_cache(self):
        rows = [self.row("a"), self.row("b")]
        metadata = []
        for row in rows:
            payload = {key: np.asarray(json.loads(row.pop(key))) for key in ("camera_intrinsics", "camera_extrinsics")}
            metadata.append(dict(payload, video_path=row["video_path"], align_factor=99))
            row["dataset_source"] = "RealEstate10K"
            row.pop("width")
        # Reverse NPZ rows to ensure joining by path rather than row number.
        metadata[1]["camera_extrinsics"][:, 0, 3] = 7
        np.savez_compressed(self.root / "train_camera.npz", np.array(metadata[::-1], dtype=object))
        self.inputs(rows)
        self.config.data.train_camera_npz = "train_camera.npz"
        self.inspect()
        dataset = RealCamMetadataDataset(self.output / "train.csv", self.root)
        self.assertEqual(dataset[0]["camera_extrinsics"][0, 0, 3], 0)
        self.assertEqual(dataset[1]["camera_extrinsics"][0, 0, 3], 7)
        self.assertEqual(dataset[0]["align_factor"], 2.5)  # CSV takes precedence
        # The downstream reader uses inspected numeric caches, not the large source NPZ.
        (self.root / "train_camera.npz").unlink()
        relocated = Path(self.temp.name) / "relocated-report"
        shutil.copytree(self.output, relocated)
        dataset = RealCamMetadataDataset(relocated / "train.csv", self.root)
        self.assertEqual(dataset[1]["camera_extrinsics"][0, 0, 3], 7)
        cache = relocated / dataset.records[0]["camera_cache"]
        np.savez_compressed(cache, camera_intrinsics=[1, 1, .5, .5],
                            camera_extrinsics=metadata[1]["camera_extrinsics"], align_factor=2.5)
        with self.assertRaisesRegex(ValueError, "Camera annotations changed"):
            dataset[0]

    def test_official_npz_missing_duplicate_and_unsupported_objects(self):
        row = self.row("a")
        metadata = {"video_path": row["video_path"], "camera_intrinsics": [1, 1, .5, .5],
                    "camera_extrinsics": np.broadcast_to(np.eye(4), (125, 4, 4)), "align_factor": 1}
        path = self.root / "train_camera.npz"
        np.savez(path, np.array([metadata, metadata], dtype=object))
        with self.assertRaisesRegex(ValueError, "Duplicate video_path"):
            load_official_camera_index(path, self.root)
        np.savez(path, np.array([dict(metadata, unsupported=Path("a"))], dtype=object))
        with self.assertRaisesRegex(ValueError, "Unsupported object"):
            load_official_camera_index(path, self.root)
        np.savez(path, np.array([metadata], dtype=object))
        self.config.data.train_camera_npz = "train_camera.npz"
        missing = self.row("not-in-npz")
        for key in ("camera_intrinsics", "camera_extrinsics"):
            missing.pop(key)
        self.inputs([missing])
        self.assertEqual(self.inspect()["rejection_reasons"], {"missing_camera_metadata": 1})
        with (self.output / "rejected.csv").open(encoding="utf-8", newline="") as handle:
            detail = next(csv.DictReader(handle))["detail"]
        self.assertIn("CSV=train.csv", detail)
        self.assertIn("camera_npz=train_camera.npz", detail)
        self.assertIn(f"video_path={missing['video_path']}", detail)

    def test_root_csv_wins_over_nested_copies_and_explicit_path_wins(self):
        root_csv = self.root / "RealEstate10K_test.csv"
        backup = self.root / "backup" / root_csv.name
        backup.parent.mkdir()
        for path in (root_csv, backup, self.root / "RealCam-Vid_test.csv"):
            path.touch()
        self.assertEqual(discover_csv(self.root, "test", subset="RealEstate10K"), root_csv)
        self.assertEqual(discover_csv(self.root, "test", "backup/RealEstate10K_test.csv",
                                     subset="RealEstate10K"), backup)
        root_csv.unlink()
        self.assertEqual(discover_csv(self.root, "test", subset="RealEstate10K"), backup)

    def test_user_csv_schema_with_auto_official_npz(self):
        rows = {}
        for split in ("train", "test"):
            path = f"RealEstate10K/{split}/source-{split}/clip.mp4"
            video = self.root / path
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
            rows[split] = [{"dataset_source": "RealEstate10K", "video_path": path,
                            "short_caption": "room", "long_caption": "a room with windows",
                            "align_factor": 2, "camera_scale": 3.5, "vtss_score": .75}]
            np.savez_compressed(self.root / f"RealCam-Vid_{split}.npz", np.array([{
                "video_path": path, "camera_intrinsics": np.array([1, 1, .5, .5]),
                "camera_extrinsics": np.broadcast_to(np.eye(4), (125, 4, 4)), "align_factor": 2}], dtype=object))
        self.inputs(rows["train"], rows["test"])
        (self.root / "train.csv").rename(self.root / "RealEstate10K_train.csv")
        self.config.data.train_csv = "RealEstate10K_train.csv"
        self.config.data.group_by = "source_or_directory"
        summary = self.inspect()
        self.assertEqual(summary["accepted"], {"train": 1, "validation": 0, "test": 1})
        sample = RealCamMetadataDataset(self.output / "train.csv", self.root)[0]
        self.assertEqual(sample["source_id"], "")
        self.assertEqual(sample["group_id"], "directory:RealEstate10K/source-train")
        self.assertEqual(sample["group_basis"], "parent_directory")
        self.assertFalse(summary["original_source_leakage_check_complete"])
        self.assertEqual(sample["short_caption"], "room")
        self.assertEqual(float(sample["camera_scale"]), 3.5)
        self.assertEqual(float(sample["vtss_score"]), .75)
        shutil.copyfile(self.root / "RealCam-Vid_train.npz", self.root / "RealEstate10K_train.npz")
        self.inspect()
        sources = json.loads((self.output / "metadata_sources.json").read_text())
        self.assertEqual(sources["RealEstate10K_train.csv"]["official_camera_npz"]["path"], "RealEstate10K_train.npz")

    def test_directory_groups_do_not_relabel_original_sources_or_metadata_split(self):
        self.config.data.group_by = "source_or_directory"
        self.config.data.validation_fraction = .5
        rows = []
        for index in range(10):
            for clip in range(2):
                path = f"RealEstate10K/train/group-{index}/clip-{clip}.mp4"
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                (self.root / path).touch()
                rows.append(self.row(f"placeholder-{index}-{clip}", video_path=path, source_video_id=""))
        # A RealCam-Vid test sample can reside in the ORIGINAL RealEstate10K train folder.
        test_path = "RealEstate10K/train/held-out-directory/clip.mp4"
        (self.root / test_path).parent.mkdir(parents=True)
        (self.root / test_path).touch()
        test_row = self.row("test-placeholder", video_path=test_path, source_video_id="")
        self.inputs(rows, [test_row])
        summary = self.inspect()
        self.assertEqual(summary["accepted"]["test"], 1)
        self.assertFalse(summary["original_source_leakage_check_complete"])
        train = RealCamMetadataDataset(self.output / "train.csv", self.root).records
        val = RealCamMetadataDataset(self.output / "validation.csv", self.root).records
        self.assertTrue(train and val)
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in val})
        self.assertTrue(all(not r["source_id"] for r in train + val))
        self.inputs(rows, [dict(test_row, video_path=rows[0]["video_path"])])
        summary = self.inspect()
        self.assertEqual(summary["overlapping_train_test_groups"], ["directory:RealEstate10K/group-0"])
        self.assertEqual(summary["rejection_reasons"]["directory_group_split_overlap"], 3)

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV is required for actual video integration")
    def test_actual_video_decode_and_broken_video(self):
        import av
        train, test = self.row("encoded"), self.row("test-encoded")
        path = self.root / train["video_path"]
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
            for index in range(125):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        shutil.copyfile(path, self.root / test["video_path"])
        observed = probe_video(path)
        self.assertEqual(observed["num_frames"], 125)
        self.assertEqual(observed["fps"], 24)
        np.testing.assert_allclose(observed["timestamps"], np.arange(125) / 24, atol=1e-6)
        self.assertEqual(probe_video(path, "header")["verification"], "header")
        self.inputs([train, self.row("broken")], [test])
        self.config.inspection.probe_mode = "decode"
        summary = self.inspect()
        self.assertEqual(summary["accepted"], {"train": 1, "validation": 0, "test": 1})
        self.assertEqual(summary["rejection_reasons"], {"video_decode_error": 1})


if __name__ == "__main__":
    unittest.main()
