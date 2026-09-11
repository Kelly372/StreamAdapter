"""Portable path conventions; no environment variables or model imports."""
from pathlib import Path, PureWindowsPath


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value, base):
    """Accept portable slash/backslash relatives and native absolute paths."""
    raw = str(value)
    if PureWindowsPath(raw).is_absolute() and not Path(raw).is_absolute():
        raise ValueError(f"Windows absolute path cannot be used on this platform: {raw}")
    path = Path(raw.replace("\\", "/")).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def workspace_root(override=None, repo_root=None):
    repo = Path(repo_root or REPO_ROOT).resolve()
    if override:
        return resolve_path(override, repo)
    # workspace/project/StreamAdapter; allow the current standalone checkout.
    if repo.parent.name.lower() == "project":
        return repo.parent.parent
    for parent in repo.parents:
        if (parent / "public_data").is_dir() and (parent / "pretrained_model").is_dir():
            return parent
    return repo.parent


def component_path(value=None, *, base_model_dir=None, filename=None):
    """Keep legacy defaults, but anchor them to the repository instead of cwd."""
    if value:
        return str(resolve_path(value, REPO_ROOT))
    root = resolve_path(base_model_dir or "wan_models/Wan2.2-TI2V-5B", REPO_ROOT)
    return str(root / filename) if filename else str(root)


def resolve_config_paths(config):
    """Opt-in workspace layout via `paths`; plain relative IO paths use repo root.

    paths.data_root/model_root are workspace-relative. Component overrides and
    checkpoints are model-root-relative when paths is present. data_path and
    eval_data_path remain repo-relative (or may use ${paths.data_root}/...).
    """
    from omegaconf import OmegaConf

    paths = config.get("paths")
    if paths is not None:
        root = workspace_root(paths.get("workspace_root"))
        paths.workspace_root = str(root)
        paths.repo_root = str(REPO_ROOT)
        paths.data_root = str(resolve_path(paths.get("data_root", "public_data/RealCam-Vid/RealEstate10K"), root))
        paths.model_root = str(resolve_path(paths.get("model_root", "pretrained_model/LongLive"), root))
        model_root = Path(paths.model_root)
        config.base_model_dir = str(resolve_path(config.get("base_model_dir", "../Wan2.2-TI2V-5B"), model_root))
    else:
        model_root = REPO_ROOT
        if config.get("base_model_dir"):
            config.base_model_dir = str(resolve_path(config.base_model_dir, REPO_ROOT))

    base = config.get("base_model_dir")
    if config.get("lightvae_path"):
        config.lightvae_path = str(resolve_path(config.lightvae_path, model_root))
    for key, filename in (("text_encoder_path", "models_t5_umt5-xxl-enc-bf16.pth"),
                          ("tokenizer_path", "google/umt5-xxl"), ("vae_path", "Wan2.2_VAE.pth")):
        if config.get(key):
            config[key] = str(resolve_path(config[key], model_root))
        elif base:
            config[key] = str(Path(base) / filename)

    for key in ("model_kwargs", "real_model_kwargs", "fake_model_kwargs"):
        kwargs = config.get(key)
        if kwargs is not None:
            if kwargs.get("model_path"):
                kwargs.model_path = str(resolve_path(kwargs.model_path, model_root))
            elif base:
                kwargs.model_path = base

    for section_name, keys, anchor in (
        ("checkpoints", ("generator_ckpt", "lora_ckpt", "real_score_ckpt", "fake_score_ckpt"), model_root),
        ("data", ("data_path", "eval_data_path", "train_csv", "test_csv"), REPO_ROOT),
        ("inference", ("output_folder",), REPO_ROOT),
    ):
        section = config.get(section_name)
        for key in keys:
            value = section.get(key) if section is not None and key in section else config.get(key)
            if value:
                resolved = str(resolve_path(value, anchor))
                config[key] = resolved
                if section is not None and key in section:
                    section[key] = resolved
    OmegaConf.resolve(config)
    return config


def add_path_arguments(parser):
    parser.add_argument("--workspace-root", default=None, help="Workspace root; otherwise inferred from repository location")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Repeatable YAML config override, e.g. --set inference.sampling_steps=4")


def load_config(path, *, workspace=None, overrides=()):
    from omegaconf import OmegaConf
    from utils.config import normalize_config

    config = OmegaConf.load(resolve_path(path, REPO_ROOT))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    if workspace is not None:
        OmegaConf.update(config, "paths.workspace_root", workspace, force_add=True)
    return normalize_config(config)
