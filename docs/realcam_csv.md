# RealEstate10K CSV 接入与检查（第 3 阶段）

本阶段读取 RealCam-Vid 的 train/test CSV，筛选 RealEstate10K，检查视频与相机标注，生成可迁移的样本清单。原视频和元数据不改写；不加载 LongLive、不开启训练。第 4 阶段再实现 24 FPS / 125 RGB 连续窗口、首帧提取和几何同步。

用户已确认训练表为 `RealEstate10K_train.csv`，包含 `dataset_source, video_path, short_caption, long_caption, align_factor, camera_scale, vtss_score`；视频路径形如 `RealEstate10K/train/<目录>/<视频>.mp4`。服务器相机文件位于数据根目录下 `RealCam-Vid_train.npz`。上一级目录只按用户描述作为相似场景分组，不能视为已验证的原始视频 ID。

本地示例 `E:/codexspace/RealCam-Vid_test.npz` 已实际读取：5,000 条记录，其中 RealEstate10K 为 2,152 条；无原始视频 ID 字段。部分 test 元数据指向 `RealEstate10K/train/...`，因此**集合划分以输入 CSV 为准，不由视频路径中的 train/test 决定**。真实 CSV 和视频尚未在本机验收。

根据 [RealCam-Vid 官方说明](https://github.com/ZGCTroy/RealCam-Vid) 和 [官方读取示例](https://github.com/ZGCTroy/RealCam-Vid/blob/main/scripts/i2v_camera_dataset.py)，内外参来自 NPZ 字典序列。[官方发布文件](https://huggingface.co/datasets/MuteApo/RealCam-Vid/tree/main) 同时提供 CSV 和 `RealCam-Vid_train.npz` / `RealCam-Vid_test.npz`。本实现按 `video_path` 关联，不能由 CSV 中的 `align_factor`、`camera_scale` 或 `vtss_score` 重建逐帧相机轨迹。

## 运行

在项目已有环境运行即可；此入口仅依赖 NumPy、OmegaConf、PyAV，不需要 Torch/GPU。独立 CPU 环境可安装：

```bash
python -m pip install numpy omegaconf av==13.1.0
```

默认配置为 `configs/data/realcam_inspection.yaml`。标准目录仍为 `workspace/project/StreamAdapter` 和 `workspace/public_data/RealCam-Vid`，无需设置环境变量。

```bash
# 每个原始 split 最多检查 10 条目标数据的视频和相机；全部行仍检查来源 ID
python inspect_realcam.py --limit 10 --tag schema-smoke

# 全量逐帧解码检查；数据量大时耗时较长
python inspect_realcam.py --tag full-decode

# 明确 CSV 相对数据根目录的位置
python inspect_realcam.py --set data.train_csv=metadata/train.csv --set data.test_csv=metadata/test.csv --tag explicit-csv

# 相机参数来自官方 NPZ（以下文件名以官方发布为例，按本地实际文件覆盖）
python inspect_realcam.py --set data.train_camera_npz=RealCam-Vid_train.npz --set data.test_camera_npz=RealCam-Vid_test.npz --limit 10 --tag npz-smoke

# 仅检查元数据和文件存在性，不验证视频可解码
python inspect_realcam.py --probe-mode metadata --limit 10 --tag metadata-smoke
```

默认 train CSV 使用已确认的 `RealEstate10K_train.csv`，test CSV 自动发现；将 `data.train_csv=null` 也可启用 train 自动发现。自动发现仅接受文件名中带独立 `train` / `test` 标记的唯一 CSV（如 `train.csv`、`RealEstate10K_train.csv`）。发现零个或多个会报错，需指定路径。

CSV 没有相机列时，会查找数据根目录下唯一的 `RealEstate10K_<split>.npz` 或 `RealCam-Vid_<split>.npz`；发现多个要求显式配置。缺失时明确报错，不生成虚拟相机。非标准布局用 `--workspace-root` 覆盖；`paths.data_root` 相对 workspace，显式 CSV/NPZ、视频和相机引用均相对数据根目录。视频、CSV 和逐样本相机文件须位于数据根目录内。官方整表 NPZ 也允许显式指定根目录外的绝对路径（如本地示例）；输出缓存可迁移，服务器配置仍应使用相对路径。视频路径同时接受 `/` 和 `\`。

## 字段约定

`data.columns` 将下列规范名映射到实际列名。例如 `--set data.columns.source_id=original_id`。值设为 `null` 可禁用某个自动映射。其他原始列不改变、不当作指令执行。

| 规范名 | 自动识别列名 | 约定 |
| --- | --- | --- |
| video_path | video_path / video | 必需，本地视频路径 |
| caption | long_caption / caption | 默认必需；可映射至 short_caption |
| subset | dataset_source / subset / dataset / data_source | 筛选 RealEstate10K；缺失时识别视频路径中的完整子集目录名 |
| source_id | source_video_id / source_id / original_video_id / youtube_id | 可选原始视频来源 ID；缺失时保留空值，按目录分组并记录检查局限；不能用 clip ID 代替 |
| clip_id | clip_id / video_id | 可选；缺失时取视频文件名，仅作记录 |
| split | split | 可选；若存在必须与输入 train/test CSV 一致 |
| camera_path | camera_path / camera_file | JSON 或数值型 NPZ；也可直接使用下面的列 |
| intrinsics | camera_intrinsics / intrinsics / K | `[4]`、`[F,4]`、`[3,3]` 或 `[F,3,3]`；四元素顺序 fx,fy,cx,cy |
| extrinsics | camera_extrinsics / extrinsics / w2c | `[F,4,4]`，与原视频逐帧对应 |
| align_factor | align_factor | 必需，显式有限正标量；此阶段只保存，不乘到坐标上 |
| short_caption / camera_scale / vtss_score | 同名列 | 可选，保留在样本清单中；不据此替代内外参或自动筛选分数 |
| valid_mask | valid_mask / camera_valid_mask | 可选，长度 F 的布尔或 0/1 数组，与计算的位姿有效性合并 |
| timestamps | timestamps / frame_timestamps | 可选，长度 F、有限、严格递增；单位在配置中声明 |
| fps | fps / video_fps | 可选，正数；解码时与实际 FPS 对照 |
| num_frames | num_frames / frame_count / video_frames | 可选，正整数；与实际视频及相机长度对照 |
| width / height | width / video_width；height / video_height | 可选，正整数；与实际视频对照 |

没有 subset 列且路径不带子集名时，只有确认整份 CSV 均为 RealEstate10K 后，才设 `data.assume_subset=RealEstate10K`。默认 `data.group_by=source_or_directory`：有明确来源 ID 时按来源分组，否则按已确认的 `<子集>/<原始split>/<目录>/<视频>` 布局提取目录分组。后者记录 `group_basis=parent_directory`、空 `source_id`，并将 `original_source_leakage_check_complete` 标为 false；不声称已排除所有原视频层面的泄漏。

若后续取得可靠的原始视频 ID，可以映射列或设置 `data.source_id_regex`（推荐捕获组 `(?P<source_id>...)`，否则取第一个组）。设置 `data.group_by=source_id` 可要求所有样本必须有明确来源 ID，缺失时剔除。

相机数组支持 JSON 列表、Python 数值列表及**未截断**的 NumPy 空格分隔数组文本，也可在内外参字段引用 `.npy` 文件。相机 JSON/数值 NPZ 内字段使用上述相机字段别名；非空 CSV 相机列优先于外部文件。JSON 中的 `.npy` 引用同样相对数据根目录。含 `...` 的截断数组及 `array(...)` 表达式不接受。

官方整表 NPZ 使用独立读取器，只允许其所需的 NumPy ndarray/dtype 等对象类型，并校验 `arr_0` 为字典数组；逐样本 `camera_path` NPZ 不使用 pickle。整表 NPZ 每个 split 需解压到内存一次，约 GB 级归档需预留足够 RAM；`--limit` 只减少视频检查和缓存数量，不能使该归档按行解压。已检查相机保存到结果目录中的数值 NPZ，后续读取不再加载整表。

默认依据官方示例声明 `camera_convention=opencv_w2c`、`intrinsics_units=normalized`、`timestamps_unit=seconds`。这些是**待真实数据核对的输入约定**，不是由矩阵数值自动推断的结论。若 CSV 已转换为 c2w / 像素内参 / 毫秒时间戳，应相应改配置。此阶段不做坐标、尺度或内参变换。

## 检查和划分

- `decode`：逐帧解码，统计实际帧数、分辨率和时间戳；校验相机长度及可用的 CSV FPS/帧数/尺寸/时间戳。只解码，不保留整段像素。
- `header`：读取视频头；未提供帧数时回退解码。无法保证全部帧可解码。
- `metadata`：检查路径存在、相机和已有数值字段；不打开视频，报告标记 `metadata_only`。
- 相机检查有限数值、齐次矩阵末行、旋转正交性/行列式、正焦距、长度和有效掩码。默认所有位姿须有效；可降低 `inspection.min_valid_pose_fraction`，保留连续有效区间供下一阶段选择。
- `eligible_target_window` 仅表示连续有效区间的时长是否足够 `(rgb_frames-1)/target_fps`；短片仍记录为 `no`，未知时为 `unknown`。它不代表已完成 125 帧对齐，也不检测切镜。
- train 的相同分组通过 `seed + group` 的稳定哈希进入同一 train/validation 分区；默认验证比例 5%，实际比例不保证精确，小样本可能为空。原 test 只保留为 test；相似场景目录分组不等于原始视频来源已验证。
- train/test 来源/目录分组重叠或视频路径重复时，剔除所有受影响的已检查行。即使设置 `--limit`，也扫描全部行的分组和路径，报告未执行相机/视频检查的数量。目录分组不包含原始路径的 train/test 层，以避免同名组被路径前缀分开。

退出码：`0` 为配置要求下审计完成（可能有剔除项，也可能是抽查/metadata 模式）；`1` 为审计未通过；`2` 为输入、配置或执行错误。无可用 train/test、来源或目录组重叠默认返回 `1`；普通坏样本剔除后继续，设 `inspection.fail_on_reject=true` 可要求零剔除。退出码 `0` 不代表数据已可训练、原视频级泄漏已排除或模型复现成功。

## 输出与下游读取

结果写入 `output/data_check_realestate10k_<mode>_seed<seed>_<时间>_<tag>/`，或用 `--run-name` 指定唯一目录；不覆盖已有目录。

| 文件 | 内容 |
| --- | --- |
| inspection.txt / resolved_config.yaml | 全部生效参数和检查摘要 |
| source_config.yaml / launch.json / status.json | 原配置、调用参数/软件版本/Git 状态、最终状态 |
| summary.json / rejected.csv | 分区与错误统计、每条剔除原因 |
| train.csv / validation.csv / test.csv | 保留的样本索引、文本、原来源和 split、视频信息、相机摘要及有效区间 |
| metadata_sources.json | 原始 CSV 相对路径、SHA256、列名映射、编码和相机约定 |
| cameras/*.npz | 使用官方整表 NPZ 时产生的逐样本相机缓存；只包含数值数组，时间戳为秒 |

索引记录原 CSV 行偏移和相机内容哈希。加载时校验原 CSV 哈希，按需读取对应行和相机文件/缓存，防止旧索引静默指向修改后的数据。官方 NPZ 的路径与哈希也记录于 `metadata_sources.json`；下游使用本次检查的缓存快照，修改原 NPZ 后需重新检查才能生效。迁移时保留整个结果子目录、数据根目录内的相对布局和原 CSV 字节内容：

```python
from utils.realcam_dataset import RealCamMetadataDataset

dataset = RealCamMetadataDataset("output/<run_name>/train.csv", "<workspace>/public_data/RealCam-Vid")
sample = dataset[0]
# sample 含 video_path、caption、source_id、camera_intrinsics、camera_extrinsics、
# align_factor、valid_mask、timestamps（秒）、camera_convention、intrinsics_units。
# 数组为 NumPy；索引 CSV 中其他标量字段目前保留为字符串。
```

这还是元数据接口，不返回视频张量，不直接改变 `run_baseline.py` 的输入格式。原版 I2V 复现仍使用图片和同名文本目录。CSV 到同步窗口/首帧的连接属于第 4 阶段。

测试：`python -m unittest discover -s tests -p test_realcam_dataset.py -v`。包含真实合成 MP4 解码，需 PyAV；未安装时该项明确跳过。
