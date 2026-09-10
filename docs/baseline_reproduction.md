# 原版 LongLive 2.0 复现

当前相机训练只完成方案配置与路径基础设施。请先通过本入口验证原版生成效果。入口使用原来的 `inference.py` 及采样算法，不加载相机 Adapter 或额外 LoRA。

## 目录

```text
workspace/
  project/StreamAdapter/
    run_baseline.py
    configs/baseline/
    example/baseline_prompts.txt
    output/
  public_data/RealCam-Vid/
  pretrained_model/
    Wan2.2-TI2V-5B/
      config.json
      diffusion_pytorch_model*.safetensors
      models_t5_umt5-xxl-enc-bf16.pth
      google/umt5-xxl/
      Wan2.2_VAE.pth
    LongLive/
      LongLive-2.0-5B/model_bf16.pt
      LongLive-2.0-5B-NVFPS-S4/model_4o6.pt
      LongLive-2.0-5B-NVFPS-S2/model_4o6.pt
```

`NVFPS` 按用户给出的本地目录命名；量化格式为 NVFP4。真实文件名/后端仍以下载的权重为准。基础 Wan 组件不能由一个 LongLive 蒸馏 checkpoint 替代。

环境安装沿用 [getting_started.md](getting_started.md) 的对应 BF16 或 NVFP4 环境。`--dry-run`/`--check-only` 只依赖 Python 和项目已有的 OmegaConf；不导入 PyTorch，不检测显存/内核。正式推理需要完整 CUDA 依赖。

## BF16：先检查，再生成

以下命令在仓库目录执行，PowerShell/Linux 都不需要 export。若从其他 cwd 启动，给 Python 的脚本参数使用脚本的实际位置；脚本内部路径仍按仓库/workspace 解析。

```text
python run_baseline.py --dry-run
python run_baseline.py --check-only
python run_baseline.py --tag first-baseline
```

- dry-run 即使发现缺失输入也会保存配置并返回成功，表示“完成配置检查”，不表示模型能运行。
- check-only 发现缺失路径时返回非零；正式运行也会先检查并停止。
- 正式运行以单进程单 GPU 为范围。不要用 torchrun 启动此 launcher；现有 SP 路径可直接使用 `inference_sp.py`，但此轮未接入其统一记录生命周期。
- 默认使用示例文本、704×1280、32 latent（125 RGB）、4 步、seed=0、BF16；不启用编译或异步 VAE，不指定第二张 GPU。
- 两步权重不能通过把四步配置中的采样步数单独改为 2 来替代。

显式指定 workspace（相对值以仓库为锚点）：

```text
python run_baseline.py --workspace-root ../.. --check-only
```

上例仅适用于标准的 `workspace/project/StreamAdapter` 布局。当前独立检出布局自动回退到仓库的父目录；可通过 `--workspace-root` 提供实际 workspace。

使用自定义提示词（路径相对仓库；一行一个样本）：

```text
python run_baseline.py --set data.data_path=example/my_prompts.txt --tag my-prompts
```

长视频测试需同时修改两个 latent 长度字段，避免噪声/输出不一致：

```text
python run_baseline.py --set num_output_frames=128 --set "data.image_or_video_shape=[1,128,48,44,80]" --tag long
```

128 latent 对应 509 RGB 帧；长度必须满足每块 8 latent 的约束。

## I2V 基线

本轮尚未实现 CSV 数据读取。先准备原版入口支持的图片与同名文本：

```text
public_data/RealCam-Vid/baseline_i2v/
  sample_001.png
  sample_001.txt
```

图片是首帧；文本是场景描述。此模式无相机数值控制。

```text
python run_baseline.py --config configs/baseline/longlive_bf16_i2v.yaml --check-only
python run_baseline.py --config configs/baseline/longlive_bf16_i2v.yaml --tag first-i2v
```

图片目录不同可用 `--set data.data_path=...`，或在 YAML 中使用 `${paths.data_root}/...`。

## NVFP4 S4 / S2

以下配置选择 FourOverSix 的 `model_4o6.pt`。需要对应 NVFP4 环境、GPU/内核支持；路径检查不验证这些条件。

```text
python run_baseline.py --config configs/baseline/longlive_nvfp4_s4.yaml --check-only
python run_baseline.py --config configs/baseline/longlive_nvfp4_s4.yaml --tag s4
python run_baseline.py --config configs/baseline/longlive_nvfp4_s2.yaml --tag s2
```

若本地目录名为 `NVFP4`：

```text
python run_baseline.py --config configs/baseline/longlive_nvfp4_s4.yaml --set checkpoints.generator_ckpt=LongLive-2.0-5B-NVFP4-S4/model_4o6.pt
```

若下载的是 Transformer Engine 的 `model_te.pt`，需同时切换权重和后端，并安装对应 TE 环境：

```text
python run_baseline.py --config configs/baseline/longlive_nvfp4_s4.yaml --set checkpoints.generator_ckpt=LongLive-2.0-5B-NVFPS-S4/model_te.pt --set model_quant_use_transformer_engine=true
```

NVFP4 I2V 可在相应 S4/S2 配置上同时覆盖三个字段：`i2v=true`、`inference.independent_first_frame=true`、`data.data_path=图片目录`。数值格式、采样步数、权重后端必须成套匹配。

## 路径规则与记录

| 参数 | 相对路径锚点 |
| --- | --- |
| `--config`、`--config_path`、`data.data_path`、`data.eval_data_path` | 仓库 |
| `--workspace-root` 的相对值 | 仓库 |
| `paths.data_root`、`paths.model_root` | workspace |
| `base_model_dir`、组件覆盖、`checkpoints.*` | 配置含 paths 时为 model_root，否则为仓库 |
| 文本编码器/Tokenizer/VAE 未单独覆盖时 | base_model_dir |
| 新复现结果 | 仓库 output 下的独立目录 |

输入配置可包含 `${paths.repo_root}`、`${paths.data_root}`、`${paths.model_root}`。绝对路径也可覆盖，但迁移时应使用原始相对配置重新运行，不直接复用旧机器的 `resolved_config.yaml`。

每次运行生成可区分的名称，例如 `baseline_t2v_longlive2_bf16_s4_f32_seed0_<时间>_<tag>`。也可使用 `--run-name`；已存在的同名目录会报错，不覆盖。

结果目录保存：

- `source_config.yaml`：输入配置；覆盖参数在 `launch.json`。
- `resolved_config.yaml`：解析后的实际配置，包括默认负面提示词、组件路径和输出目录。
- `inference.txt`：参数文本；模型初始化成功后追加运行时默认参数。
- `runtime_config.yaml`：模型管线/调度器参数与 GPU 信息（实际初始化成功后生成）。
- `launch.json`：参数数组、代码提交和环境版本；`input_check.json`：输入检查。
- `status.json`、`console.log`、生成的视频/提示词文件。

默认负面提示词由项目配置工具填入。更深层的管线/调度器默认值在实际创建模型后记录；dry-run 不能声称已获得 GPU 运行时信息。

本轮尚未完成相机训练，因此 `configs/camera_adapter/experiment_a.yaml` 不能用于启动训练。完整路线与后续状态见 [code_log.md](../code_log.md)。
