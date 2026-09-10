"""CPU-only tests for step 1/2; no PyTorch or model weights required."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
from utils import project_paths
from utils.config import normalize_config
from utils.project_paths import REPO_ROOT, component_path, load_config, resolve_path, workspace_root
import run_baseline
from utils.run_record import save_runtime_record


class ProjectPathsTest(unittest.TestCase):
    def test_layout_and_explicit_override(self):
        with tempfile.TemporaryDirectory(prefix="workspace with spaces ") as directory:
            root = Path(directory).resolve()
            repo = root / "project" / "StreamAdapter"
            self.assertEqual(workspace_root(repo_root=repo), root)
            self.assertEqual(workspace_root("../..", repo_root=repo), root)
            self.assertEqual(workspace_root(repo_root=root / "standalone"), root)

    def test_baseline_relocation_and_component_separation(self):
        for name in ("machine_a", "machine_b with spaces"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix=name) as directory:
                root = Path(directory).resolve()
                config = load_config("configs/baseline/longlive_bf16.yaml", workspace=str(root))
                self.assertEqual(Path(config.base_model_dir), root / "pretrained_model/Wan2.2-TI2V-5B")
                self.assertEqual(Path(config.generator_ckpt), root / "pretrained_model/LongLive/LongLive-2.0-5B/model_bf16.pt")
                self.assertEqual(Path(config.tokenizer_path), Path(config.base_model_dir) / "google/umt5-xxl")
                self.assertEqual(config.model_kwargs.model_path, config.base_model_dir)

    def test_paths_do_not_depend_on_cwd(self):
        expected = load_config("configs/baseline/longlive_bf16.yaml")
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = load_config("configs/baseline/longlive_bf16.yaml")
                self.assertEqual(config, expected)
                self.assertEqual(Path(component_path()), REPO_ROOT / "wan_models/Wan2.2-TI2V-5B")
            finally:
                os.chdir(old_cwd)

    def test_backslash_and_spaces(self):
        self.assertEqual(resolve_path(r"public_data\a b\clip.mp4", REPO_ROOT),
                         (REPO_ROOT / "public_data/a b/clip.mp4").resolve())

    def test_overrides_and_component_paths(self):
        config = load_config("configs/baseline/longlive_bf16.yaml", overrides=[
            "paths.model_root=pretrained_model/custom", "base_model_dir=../Wan2.2-TI2V-5B",
            "text_encoder_path=custom_t5.pth", "checkpoints.generator_ckpt=merged/model.pt",
            "inference.sampling_steps=2", "logging.seed=42",
        ])
        self.assertEqual(config.sampling_steps, 2)
        self.assertEqual(config.seed, 42)
        self.assertEqual(Path(config.text_encoder_path), Path(config.paths.model_root) / "custom_t5.pth")
        self.assertEqual(Path(config.generator_ckpt), Path(config.paths.model_root) / "merged/model.pt")
        self.assertEqual(config.generator_ckpt, config.checkpoints.generator_ckpt)

    def test_all_configs_normalize_and_roundtrip(self):
        for path in (REPO_ROOT / "configs").rglob("*.yaml"):
            with self.subTest(config=path.name):
                config = load_config(path)
                restored = normalize_config(OmegaConf.create(OmegaConf.to_yaml(config, resolve=True)))
                self.assertEqual(OmegaConf.to_container(config), OmegaConf.to_container(restored))

    def test_experiment_a_contract_and_not_runnable(self):
        config = load_config("configs/camera_adapter/experiment_a.yaml")
        self.assertEqual(config.experiment.status, "spec_only")
        self.assertEqual(config.rgb_frames, 4 * (config.image_or_video_shape[1] - 1) + 1)
        self.assertEqual(config.algorithm.target_gradient, "full_sampling_chain")
        self.assertFalse(config.algorithm.load_real_score)
        self.assertFalse(config.algorithm.load_fake_score)
        self.assertEqual(config.loss.type, "masked_latent_l1")
        for stage in config.curriculum.stages:
            self.assertAlmostEqual(sum(stage.probabilities), 1.0)
            self.assertIsNone(stage.steps)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            run_baseline.validate_baseline(config)

    def test_inference_profiles_match_steps_and_backend(self):
        for suffix, steps in (("s4", 4), ("s2", 2)):
            config = load_config(f"configs/baseline/longlive_nvfp4_{suffix}.yaml")
            self.assertTrue(config.model_quant)
            self.assertFalse(config.model_quant_use_transformer_engine)
            self.assertEqual(config.sampling_steps, steps)
            self.assertIn(f"NVFPS-{suffix.upper()}", config.generator_ckpt)
            run_baseline.validate_baseline(config)
            i2v_config = load_config(f"configs/baseline/longlive_nvfp4_{suffix}_i2v.yaml")
            self.assertTrue(i2v_config.i2v)
            self.assertTrue(i2v_config.independent_first_frame)
            self.assertEqual(i2v_config.sampling_steps, steps)
            self.assertEqual(i2v_config.generator_ckpt, config.generator_ckpt)
            run_baseline.validate_baseline(i2v_config)
        config = load_config("configs/baseline/longlive_bf16_i2v.yaml")
        self.assertTrue(config.i2v)
        self.assertTrue(config.independent_first_frame)
        self.assertEqual(Path(config.data_path), Path(config.paths.data_root) / "baseline_i2v")

    def test_invalid_length_is_rejected(self):
        config = load_config("configs/baseline/longlive_bf16.yaml", overrides=["num_output_frames=64"])
        with self.assertRaisesRegex(ValueError, "block-aligned"):
            run_baseline.validate_baseline(config)

    def test_complete_input_fixture_and_missing_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/baseline/longlive_bf16.yaml", workspace=directory)
            for key in ("base_model_dir", "tokenizer_path"):
                Path(config[key]).mkdir(parents=True, exist_ok=True)
            for key in ("text_encoder_path", "vae_path", "generator_ckpt"):
                path = Path(config[key])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (Path(config.base_model_dir) / "config.json").write_text("{}")
            (Path(config.base_model_dir) / "diffusion_pytorch_model.safetensors").touch()
            self.assertEqual(run_baseline.check_inputs(config), [])
            Path(config.generator_ckpt).unlink()
            self.assertTrue(any("generator_ckpt" in error for error in run_baseline.check_inputs(config)))

    def test_dry_run_records_without_gpu_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_repo = Path(directory)
            profile = Path("configs/baseline/longlive_bf16_i2v.yaml")
            (temp_repo / profile).parent.mkdir(parents=True)
            (temp_repo / profile).write_text((REPO_ROOT / profile).read_text(encoding="utf-8"), encoding="utf-8")
            argv = ["run_baseline.py", "--dry-run", "--run-name", "test-run"]
            with patch.object(run_baseline, "REPO_ROOT", temp_repo), patch("sys.argv", argv), \
                    patch.object(project_paths, "REPO_ROOT", temp_repo), \
                    patch.object(run_baseline, "environment_record", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_baseline.main(), 0)
                folder = temp_repo / "output/test-run"
                config = OmegaConf.load(folder / "resolved_config.yaml")
                self.assertEqual(config.output_folder, str(folder))
                self.assertTrue(config.i2v)
                self.assertTrue(config.independent_first_frame)
                self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "dry_run")
                self.assertTrue((folder / "inference.txt").is_file())
                self.assertTrue((folder / "source_config.yaml").is_file())
                with self.assertRaises(FileExistsError):
                    run_baseline.main()

    def test_check_only_missing_inputs_returns_failure(self):
        source = str(REPO_ROOT / "configs/baseline/longlive_bf16.yaml")
        with tempfile.TemporaryDirectory() as directory:
            argv = ["run_baseline.py", "--config", source, "--check-only", "--workspace-root", directory]
            with patch.object(run_baseline, "REPO_ROOT", Path(directory)), patch("sys.argv", argv), \
                    patch.object(run_baseline, "environment_record", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_baseline.main(), 1)

    def test_runtime_defaults_record_with_library_config_types(self):
        class TensorValues:
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [1000.0, 500.0]

        fake_torch = SimpleNamespace(
            empty=lambda *args, **kwargs: None,
            cuda=SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(name="test gpu", total_memory=1024)),
            version=SimpleNamespace(cuda="test"),
        )
        pipeline = SimpleNamespace(
            sampling_steps=2, sample_solver="unipc",
            _dit_model=SimpleNamespace(config={"patch_size": (1, 2, 2)}),
            _initialize_sample_scheduler=lambda noise: SimpleNamespace(
                config={"solver": "test", "defaults": {"a", "b"}},
                timesteps=TensorValues(), sigmas=TensorValues()),
        )
        with tempfile.TemporaryDirectory() as directory:
            config = OmegaConf.create({"output_folder": directory})
            with patch.dict("sys.modules", {"torch": fake_torch}), \
                    patch("utils.run_record.environment_record", return_value={}):
                save_runtime_record(config, pipeline, "cuda:0", 1)
            saved = OmegaConf.load(Path(directory) / "runtime_config.yaml")
            self.assertEqual(saved.pipeline_parameters.sampling_steps, 2)
            self.assertEqual(list(saved.model_configuration.patch_size), [1, 2, 2])
            self.assertEqual(list(saved.scheduler.defaults), ["a", "b"])
            self.assertIn("Runtime parameters", (Path(directory) / "inference.txt").read_text())


if __name__ == "__main__":
    unittest.main()
