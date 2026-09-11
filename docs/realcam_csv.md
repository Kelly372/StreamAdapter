# RealEstate10K CSV 接入与检查（第 3 阶段）

本阶段读取 RealCam-Vid 的 train/test CSV，筛选 RealEstate10K，检查视频与相机标注，生成可迁移的样本清单。原视频和元数据不改写；不加载 LongLive、不开启训练。第 4 阶段通过 `prepare_realcam_windows.py` 实现 24 FPS / 125 RGB 连续窗口、首帧提取和几何同步。

当前数据根目录为 `workspace/public_data/RealCam-Vid/RealEstate10K`。用户手动整理 train/test CSV 并将 video_path 改为 `train/<目录>/<视频>.mp4` 或 `test/<目录>/<视频>.mp4`；代码不修改 CSV。字段为 `dataset_source, video_path, short_caption, long_caption, align_factor, camera_scale, vtss_score`。CSV 和官方 NPZ 均使用 `dataset_source=RealEstate10K`；`data_source` 仅为兼容别名。上一级视频目录只作为相似场景分组，不视为已验证的原始视频 ID。

```text
workspace/public_data/RealCam-Vid/
  RealCam-Vid_train.npz
  RealCam-Vid_test.npz
  RealEstate10K/
    RealEstate10K_train.csv
    RealEstate10K_test.csv
    RealEstate10K_train.npz  # 新 3A 按 CSV 重建后，由用户移动至此
    RealEstate10K_test.npz
    train/<sub_dir>/<video_name>.mp4
    test/<sub_dir>/<video_name>.mp4
```

例如 `video_path=train/ZPLUfZsgEtg/f3fa5c1e24a522bc.mp4`，拼接数据根目录得到 `workspace/public_data/RealCam-Vid/RealEstate10K/train/ZPLUfZsgEtg/f3fa5c1e24a522bc.mp4`。支持 `./train/...`、Windows 分隔符及位于数据根目录内的绝对路径；相对路径更适合迁移。文本读取同一行的 long_caption，CSV 中非空 align_factor 优先于 NPZ。

本地示例 `E:/codexspace/RealCam-Vid_test.npz` 已实际读取：5,000 条记录，其中 RealEstate10K 为 2,152 条；无原始视频 ID 字段。部分 test 元数据指向 `RealEstate10K/train/...`，因此**集合划分以输入 CSV 为准，不由视频路径中的 train/test 决定**。真实 CSV 和视频尚未在本机验收。

根据 [RealCam-Vid 官方说明](https://github.com/ZGCTroy/RealCam-Vid) 和 [官方读取示例](https://github.com/ZGCTroy/RealCam-Vid/blob/main/scripts/i2v_camera_dataset.py)，内外参来自 NPZ 字典序列。[官方发布文件](https://huggingface.co/datasets/MuteApo/RealCam-Vid/tree/main) 同时提供 CSV 和 `RealCam-Vid_train.npz` / `RealCam-Vid_test.npz`。本实现按 `video_path` 关联，不能由 CSV 中的 `align_factor`、`camera_scale` 或 `vtss_score` 重建逐帧相机轨迹。

## 运行

第 3 步成功（退出码 0）后，在常规摘要末尾打印第 4 步命令，自动引用本次 `output/<run_name>/train.csv`，默认 `--limit 10 --export-count 2`。例如 `realestate10k_audit_smoke_v2` 会建议输出到 `realestate10k_windows_smoke_v2`；已有同名目录时自动追加编号。非默认数据根目录、workspace、seed、目标帧数/FPS 会传给下一步。失败时不打印继续命令。

提示命令应在仓库目录执行，也保存在本次结果的 `next_command.txt` 和 `next_command.json`（参数数组、工作目录和 shell）。命令仅供复制，不自动执行。全量检查成功后仍先建议抽查少量窗口，确认后可另建全量索引。

在项目已有环境运行即可；此入口仅依赖 NumPy、OmegaConf、PyAV，不需要 Torch/GPU。独立 CPU 环境可安装：

```bash
python -m pip install numpy omegaconf av==13.1.0
```

默认配置为 `configs/data/realcam_inspection.yaml`。仓库位于 `workspace/project/StreamAdapter`，数据位于 `workspace/public_data/RealCam-Vid/RealEstate10K`，无需设置环境变量。

```bash
# 每个原始 split 最多检查 10 条目标数据的视频和相机；全部行仍检查来源 ID
python inspect_realcam.py --limit 10 --tag schema-smoke

# 全量逐帧解码检查；数据量大时耗时较长
python inspect_realcam.py --tag full-decode

# 明确 CSV 相对数据根目录的位置
python inspect_realcam.py --set data.train_csv=metadata/train.csv --set data.test_csv=metadata/test.csv --tag explicit-csv

# 明确使用按 CSV 重建、已移动到数据根目录的 NPZ
python inspect_realcam.py --set data.train_camera_npz=RealEstate10K_train.npz --set data.test_camera_npz=RealEstate10K_test.npz --limit 10 --tag npz-smoke

# 仅检查元数据和文件存在性，不验证视频可解码
python inspect_realcam.py --probe-mode metadata --limit 10 --tag metadata-smoke
```

默认 train/test CSV 自动发现，优先选择数据根目录直接包含的 `RealEstate10K_<split>.csv`，也兼容 `RealState10K_<split>.csv` 拼写。根目录子集表存在时不递归搜索，避免备份或导出的同名 CSV 干扰。根目录没有子集表时才递归查找，仍优先子集表，再接受文件名中带独立 train/test 标记的唯一 CSV。同一优先级多个候选（包括两种拼写同时存在）仍要求显式设置 `data.train_csv/test_csv`；显式路径始终优先。

CSV 没有相机列时，优先选择数据根目录下的 `RealEstate10K_<split>.npz`，缺失时使用 `RealCam-Vid_<split>.npz`；显式配置优先于自动发现。两者都缺失时明确报错，不生成虚拟相机。非标准布局用 `--workspace-root` 覆盖；`paths.data_root` 相对 workspace，显式 CSV/NPZ、视频和相机引用均相对数据根目录。视频、CSV 和逐样本相机文件须位于数据根目录内。官方整表 NPZ 也允许显式指定根目录外的绝对路径（如本地示例）；输出缓存可迁移，服务器配置仍应使用相对路径。视频路径同时接受 `/` 和 `\`。

## 第 3A 步：预先分离 RealEstate10K NPZ

### 先查看原始 NPZ（不筛选、不分离）

```bash
python extract_realestate10k.py --inspect-only
```

此模式默认只加载 `--source-root`（默认 `workspace/public_data/RealCam-Vid`）中的 `RealCam-Vid_test.npz`，打印前 3 条记录的路径/数组形状，以及全表数据来源、路径前缀统计。保留原文件所有子集、字段、顺序、路径和数组数值，不读取 CSV 或视频，也不修改 NPZ。

可用 `--test-npz <文件路径>` 覆盖输入；相对路径基于 source-root。`--preview-count 10` 控制终端摘要和展开预览的条数，不限制完整导出的记录数量。查看两份归档可加 `--splits train test`。

结果位于 `output/metadata_readable_test_<时间>_<tag>/`（未设置 tag 时省略该后缀），其中 `test/` 包含：

| 文件 | 用途 |
| --- | --- |
| summary.txt / summary.json | 全表数量、dataset_source 分布、路径前缀分布、各字段类型及数组形状/dtype 统计 |
| index.jsonl | 每行一条记录，列出原始序号、来源、完整路径、数组形状，适合搜索视频路径 |
| preview.json | 前若干条完整记录，缩进展开，适合在编辑器中人工查看相机矩阵 |
| records.jsonl | 所有记录的全部字段和数组数值，每行一条，无省略号截断；文件可能较大 |

数组表示为 `{"__ndarray__": true, "dtype": "float64", "shape": [F,4,4], "values": [...]}`；特殊 NaN/Infinity 使用显式标记对象，保证标准 JSON 可读。此格式用于人工查看，不替代原 NPZ。输出根目录保留 extraction.txt、parameters.json、launch.json、summary.json、status.json；`status=completed` 表示导出完成，不表示第三步的数据检查通过。

### 临时匹配诊断

遇到 `missing_camera_metadata` 时，可先运行临时只读诊断（不是第三步的替代）：

```bash
python diagnose_realcam_temp.py --audit-dir output/realestate10k_audit_smoke_v2 --splits test --limit 10
```

默认沿用上次检查的 resolved_config.yaml 和 metadata_sources.json 中实际选择的 NPZ，并优先检查 rejected.csv 的路径。没有历史输出时，可用 `--camera-metadata-dir output/realestate10k_metadata_v2` 替代 `--audit-dir`；`--config`、`--workspace-root` 和 `--set` 可覆盖配置。诊断双集合可省略 `--splits test`。

每次生成唯一的 `output/diagnose_realcam_temp_<时间>/`，保存 diagnosis.txt、report.json、参数、配置和状态；输入文件不修改。匹配失败会自动对照数据根目录及所选 NPZ 目录内可用的 train/test 子集和总表；`--compare-full` 可在匹配成功时也强制对照。NPZ 顺序加载，每份完整解压到内存一次，仍需足够 RAM；不会解码视频，也不加载模型。

报告区分完整路径精确匹配、仅大小写不同、场景目录＋文件名相同、仅文件名相同。后几类仅为排错线索，不替换正式读取的精确匹配。退出码 0 仅表示所抽查的路径在选定 NPZ 中存在、视频文件存在且文本非空，仍需重新执行第三步验证解码和相机几何；退出码 1 表示匹配/文件问题，2 表示诊断执行失败。分享本次 diagnosis.txt 和 report.json 即可进一步定位。

服务器问题定位并验证后，可删除 `diagnose_realcam_temp.py`、`tests/test_diagnose_realcam_temp.py` 和本段临时说明；生产入口不依赖该脚本。

相机按完整、规范化后的相对 `video_path` 精确匹配，不按行号或视频文件名匹配，不自动替换路径中的 train/test。匹配失败时 `rejected.csv` 同时记录该路径、输入 CSV、所选 NPZ 和实际视频文件路径。目录或字段名适配不能补齐 NPZ 中确实缺失的条目。更新输入 CSV/NPZ 或选择规则后应重新运行第 3 步，并以新清单重建第 4 步窗口索引。

### 分离子集 NPZ

`extract_realestate10k.py` 默认 `--split-policy csv`：读取两份原始 NPZ，按用户已整理的 CSV 样本与顺序重建对应 train/test NPZ。原始归档名和视频路径中的 train/test 都不决定新集合归属。相机只按完整路径精确关联，原始 NPZ 路径仅移除 RealEstate10K/ 前缀；CSV 不改写。

在标准服务器目录中运行：

```bash
python extract_realestate10k.py --run-name realestate10k_csv_aligned_v1
```

默认读取 `workspace/public_data/RealCam-Vid/RealCam-Vid_train.npz` 和 `RealCam-Vid_test.npz`，在仓库生成：

```text
output/realestate10k_csv_aligned_v1/
  RealEstate10K_train.npz
  RealEstate10K_test.npz
  partition_report.json
  train_provenance.json
  test_provenance.json
  extraction.txt
  parameters.json
  launch.json
  summary.json
  status.json
```

检查 CSV 的来源字段、短路径格式、重复路径及跨集合重复。两份源 NPZ 中同一路径的元数据若在路径规范化后全部一致则合并并记录两份来源；其余任何字段差异（含数组 dtype/形状/数值）均视为冲突。缺失匹配或冲突时返回非零，并在 partition_report.json 记录原因，此时不生成子集 NPZ。

成功后重新加载每份输出，逐字段验证，并检查路径集合和顺序与对应 CSV 完全一致。所有相机数组保留原始可变长度，不补齐、不截断；除 video_path 外原字段不变。provenance 文件记录每条 CSV 行对应的源 NPZ。原文件及已有输出目录不覆盖；重复执行使用新 --run-name 或 --tag。

原始 NPZ 逐份加载；命中的记录暂存于磁盘 SQLite，避免同时保留两份大归档的数组，结束时删除临时数据库。每份源/输出 NPZ 仍需完整解压，需预留内存与磁盘空间。此步不解码视频、不验证相机几何，成功后仍运行第三步。

随后直接让检查器读取输出目录中的 NPZ，无需复制到原数据目录：

```bash
python inspect_realcam.py --camera-metadata-dir output/realestate10k_csv_aligned_v1 --limit 10 --tag subset-smoke
```

`--camera-metadata-dir` 相对仓库根目录解析，也接受绝对路径；此参数会覆盖两个 `data.*_camera_npz` 设置，并要求所选文件存在，不静默回退总 NPZ。CSV 和视频仍从原数据根目录读取。全量检查时移除 `--limit 10`。

用户将两份新 NPZ 移动到 RealEstate10K 文件夹后，可直接运行 `python inspect_realcam.py --limit 10 --tag relocated-smoke`。请保留输出目录中的参数与来源报告。第 3A 步不会移动或修改 CSV。

保留的旧模式仅按原归档筛选，不对齐 CSV，也不缩短路径（用于对照，不适用于新默认目录）：

```bash
python extract_realestate10k.py --split-policy archive --splits test --test-npz "E:/codexspace/RealCam-Vid_test.npz" --tag legacy-test
```

`--source-root` 和 `--data-root` 均相对 workspace；前者默认 public_data/RealCam-Vid，后者默认 public_data/RealCam-Vid/RealEstate10K。`--train-npz` / `--test-npz` 相对 source-root，`--train-csv` / `--test-csv` 相对 data-root，均可用绝对路径覆盖。新模式即使只指定 `--splits test`，也会读取两份源 NPZ；默认生成两份输出并检查 CSV 间重复。成功后打印使用本次输出的第三步命令。旧索引绑定旧 CSV 哈希，修改/迁移后必须重建第三、四步输出。

## 字段约定

`data.columns` 将下列规范名映射到实际列名。例如 `--set data.columns.source_id=original_id`。值设为 `null` 可禁用某个自动映射。其他原始列不改变、不当作指令执行。

| 规范名 | 自动识别列名 | 约定 |
| --- | --- | --- |
| video_path | video_path / video | 必需，本地视频路径 |
| caption | long_caption / caption | 默认必需；可映射至 short_caption |
| subset | dataset_source / data_source / subset / dataset | 筛选 RealEstate10K；缺失时识别视频路径中的完整子集目录名；多列存在时按此顺序，或用 data.columns.subset 显式指定 |
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

第三步没有 subset 列且路径不带子集名时，只有确认整份 CSV 均为 RealEstate10K 后，才设 `data.assume_subset=RealEstate10K`；新第 3A 步要求 CSV 显式提供来源字段。默认 `data.group_by=source_or_directory`：有明确来源 ID 时按来源分组，否则支持 `<原始split>/<目录>/<视频>`，并兼容显式旧根目录下的 `<子集>/<原始split>/<目录>/<视频>`。后者记录 `group_basis=parent_directory`、空 source_id，原视频级泄漏检查仍标为未完成。

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

以上仍是第 3 步元数据接口；视频窗口和首帧由已实现的 [第 4 步窗口加载器](realcam_windows.md) 读取/导出。原版 `run_baseline.py` 继续使用图片和同名文本目录，可直接读取第 4 步输出的 baseline_i2v/。

测试：`python -m unittest discover -s tests -p test_realcam_dataset.py -v`。包含真实合成 MP4 解码，需 PyAV；未安装时该项明确跳过。
