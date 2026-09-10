# StreamAdapter 代码修改与实施记录

本文件是本项目的持续维护清单。后续每次实施应同步更新任务状态、修改文件、验证结果、限制和下一步；只有配置定义完成时，不将对应训练功能标记为已实现。

## 全局约定

- **项目目标固定为 I2V：首帧图像＋文本＋相机轨迹 → 受控视频。基础复现也默认 I2V；T2V 仅作可选对照。**

- 目标目录：`workspace/project/StreamAdapter`。当前本地检出的目录名可以不同。
- 数据：`workspace/public_data/RealCam-Vid` 中的 RealEstate10K；元数据为 train/test CSV。
- 已确认训练表 `RealEstate10K_train.csv`，字段为 `dataset_source, video_path, short_caption, long_caption, align_factor, camera_scale, vtss_score`。`dataset_source=RealEstate10K`，视频路径形如 `RealEstate10K/train/<目录>/<视频>.mp4`；相机参数来自数据根目录的 `RealCam-Vid_train.npz`。上一级目录为相似场景分组，不标记为已验证的原始视频 ID。
- Wan2.2 基础组件：`workspace/pretrained_model/Wan2.2-TI2V-5B`。
- LongLive 蒸馏权重：`workspace/pretrained_model/LongLive` 下的 `LongLive-2.0-5B`、`LongLive-2.0-5B-NVFPS-S4`、`LongLive-2.0-5B-NVFPS-S2`。
- `NVFPS` 是用户提供的本地目录拼写，数值格式仍为 NVFP4；实际目录若为 `NVFP4`，使用配置/命令行覆盖。
- 相对目录结构保持稳定，绝对路径允许变化；使用配置和命令行，尽量不要求手动 export。
- 结果放在仓库 `output/<有区分性的运行名称>/`；保存最终生效参数（含默认值）、启动信息、代码版本和数据/权重路径。训练记录为 `training.txt`，推理为 `inference.txt`。
- 原始相对配置与本次解析的绝对路径分别保存，便于迁移和审计。

## 已确定的首轮方案 A

冻结已蒸馏 LongLive 2.0 / Wan2.2-TI2V-5B、VAE、文本编码器，仅训练相机 Adapter。

- 输入：首帧、文本、Plücker Ray；采用相机残差条件分支。
- 数据：24 FPS、125 RGB 帧，对应 32 latent 帧、4 个 8-latent 块。首帧属于第一个块，不额外添加为第 33 个 latent。
- 历史：当前模型生成；停止梯度；不以 GT 后续视频替换历史。
- 每轮抽一个目标块，目标块完整少步采样链保留梯度；对最终 latent 计算与 GT 的 L1。
- 损失排除固定首帧和无效/填充位置，按有效 latent 元素平均。
- 参数更新后丢弃本轮 KV/历史，下一轮重新生成。
- 不加载 teacher/fake；不加入 FM、时间差分、边界或相机几何损失。
- 相机坐标在整个窗口起点统一，各块不重新归零。

| 阶段 | 块 1 | 块 2 | 块 3 | 块 4 |
| --- | ---: | ---: | ---: | ---: |
| 预热 | 100% | 0 | 0 | 0 |
| 加入续接 | 50% | 50% | 0 | 0 |
| 扩展历史 | 30% | 30% | 40% | 0 |
| 四块混合 | 25% | 25% | 25% | 25% |

Adapter 宽度/深度/注入层、阶段长度、训练预算尚未定稿。首轮配置使用 BF16/4 步作为实施起点；不同少步数必须匹配对应蒸馏权重。效果尚待实验验证。

## 完整修改清单

### 1. 固定首轮实验配置 — 已完成配置定义

- [x] 新增独立配置，记录冻结范围、相机输入、125→32→4、历史来源、目标梯度、损失和课程概率。
- [x] 与官方 AR/DMD/LoRA 配置区分；尚未确定的结构/训练参数保留 null。
- [x] 标为 `spec_only`，在原训练/推理入口防止误启动。
- 文件：`configs/camera_adapter/experiment_a.yaml`。
- 此状态不代表相机 Adapter 或训练器已实现。

### 2. 统一路径与运行目录 — 已完成首轮接入，待目标 GPU 环境验收

- [x] 自动从 `workspace/project/StreamAdapter` 推导 workspace，支持 `--workspace-root` 覆盖。
- [x] 统一相对路径解析，处理 Windows/Linux 分隔符；避免依赖终端 cwd。
- [x] 数据根目录与基础组件/蒸馏权重路径分离。
- [x] 主干、文本编码器、Tokenizer、VAE 路径接入；旧配置组件默认值锚定仓库。
- [x] train/inference/inference_sp 支持 `--set KEY=VALUE` 和 workspace 覆盖。
- [x] 提供原版 BF16 I2V、NVFP4 S4/S2 I2V 复现配置与默认 I2V 的 `run_baseline.py`；保留显式 T2V 对照。
- [x] 复现入口启动前检查文件，支持无 GPU 的 dry-run/check-only。
- [x] 复现结果存入 `output/<run_name>/`，同名运行拒绝覆盖，并保存参数/状态。
- [ ] 在用户实际模型和 CUDA 环境执行基础模型推理，确认视频输出。
- 当前统一记录入口为 `run_baseline.py`（单进程单卡）；原始 train/inference_sp 的完整运行目录生命周期将在第 12 步接入。

### 3. RealCam-Vid CSV 数据接入与检查 — 已实现首轮，待真实数据验收

- [x] 接入用户确认的 CSV 列名/相对视频路径，读取 train/test 并筛选 RealEstate10K；支持列名映射。
- [x] 按 video_path 关联官方相机 NPZ；另支持列内数组、JSON/数值 NPZ/内外参 NPY 引用。保留相机坐标及 align_factor，不提前做几何转换。
- [x] 检查文件、FPS、实际解码帧数、位姿长度/有效性、可用时间戳，保存剔除原因与统计；明确区分 metadata/header/decode。
- [x] 保留 CSV 原始 train/test；验证集从 train 按明确原视频 ID 或场景目录稳定分组，分别记录分组依据。检查重复路径及组间重叠，抽查时也扫描完整分组；无真实来源 ID 时不宣称完成原视频级泄漏检查。
- [x] 保留原始视频，提供可迁移清单和按需相机读取接口；实际窗口索引与视频张量采样在第 4 步实现。
- [x] 独立 output 子目录记录全部配置、inspection.txt、数据哈希/字段映射、运行环境、清单和状态；拒绝同名覆盖。
- [x] 已读取本地实际 `E:/codexspace/RealCam-Vid_test.npz`，核实字段、数组形状及部分 test 条目指向原始 train 视频目录；不能从路径推断 RealCam-Vid 的集合划分。
- [ ] 在用户真实 CSV 和视频上执行完整验收；补充可靠原视频 ID 后可进一步核实来源级泄漏。当前只承诺目录分组不跨 train/validation，重叠 test 分组被剔除并报告。
- 文件：`inspect_realcam.py`、`utils/realcam_dataset.py`、`utils/realcam_inspection.py`、`configs/data/realcam_inspection.yaml`、`docs/realcam_csv.md`。

### 4. 连续窗口采样与几何同步 — 待实现

- [ ] 按 24 FPS 的时间间隔随机采样 125 帧，RGB/相机使用相同实际索引。
- [ ] 首帧取当前窗口首帧；裁剪/缩放同步更新内参。
- [ ] 关闭没有几何同步的增强；不跨切镜拼接，不复制/循环补监督。
- [ ] 返回来源 ID、索引、时间戳和有效掩码。

### 5. 相机转换与 Plücker 编码 — 待实现

- [ ] 核实内参单位、轴向、w2c/c2w、align_factor，再统一转换。
- [ ] 整个窗口一次性建立首帧参考系，不按块重置，不逐帧归一化尺度。
- [ ] 生成六通道 Plücker 条件，固定通道/坐标/尺度约定。
- [ ] 对齐 RGB、latent 和 token 时间/空间网格；特殊处理首帧，避免未来块泄漏。
- 建议：`utils/camera_geometry.py`。

### 6. 相机 Adapter 与主干注入 — 待实现

- [ ] 新增相机特征编码与残差分支，输出投影零初始化。
- [ ] 宽度、深度、注入层、控制强度配置化，与 LoRA adapter 配置区分。
- [ ] wrapper/主干传递相机条件，按块切片；关闭分支时保持基线行为。
- 涉及：`utils/wan_5b_wrapper.py`、`wan_5b/modules/causal_model.py`；建议新增 `model/camera_adapter.py`。

### 7. 仅加载生成器的相机训练入口 — 待实现

- [ ] 只加载生成器、VAE、文本编码器和相机 Adapter；不继承会额外构建 real/fake 的初始化。
- [ ] 冻结基础组件，优化器仅包含 Adapter，保存可训练参数列表/数量。
- [ ] 接入数据、精度、梯度累积/裁剪、验证。
- 建议新增 `train_camera_adapter.py`、`trainer/camera_adapter.py`、`model/camera_supervised.py`。

### 8. 可微分块少步采样 — 待实现

- [ ] 抽取目标块 k，无梯度生成前 k−1 块；历史 latent/KV detach。
- [ ] 目标块完整执行真实推理调度并保留所有步骤的计算图；每步固定 I2V 首帧。
- [ ] 检查 KV 原地更新、RoPE、注意力、调度器的梯度支持，避免破坏反向传播。
- [ ] 同次展开保留块间历史；参数更新后彻底丢弃历史。
- [ ] 训练和推理共享调度规则；不直接复用随机退出/截断去噪链的 Self-Forcing 训练行为。

### 9. 分阶段目标块抽样 — 待实现

- [ ] 实现本文件所列四阶段概率，配置阶段长度/切换方式。
- [ ] 记录实际抽样、阶段和全局步数，断点恢复这些状态。
- [ ] 多卡版本需要正确同步相关采样状态；不把阶段切换等同于 loss 达标。

### 10. Masked latent L1 — 待实现

- [ ] 冻结 VAE 编码 GT，统一布局/归一化。
- [ ] 仅监督目标块最终 latent；排除固定首帧和填充，按有效元素平均。
- [ ] 分块记录误差；首块 7 个可监督 latent 时间位置，其余各 8 个。

### 11. 相机推理与验证 — 待实现

- [ ] 加载基础模型+Adapter，支持首帧/文本/相机轨迹和 test CSV 选样。
- [ ] 复用训练几何与采样；保存 GT/输出、逐块误差和完整样本元信息。
- [ ] 固定首帧/噪声改变轨迹；Adapter 开关对照；检查跨块跳变、模糊、旧块退化。
- [ ] 无对应 GT 的新轨迹只用于控制响应检查，不计算对原轨迹视频的配对指标。

### 12. 完整结果与断点管理 — 部分前置到第 2 步

- [x] 基础复现支持独立 output 子目录、原配置/生效配置/参数文本/启动环境/状态记录。
- [x] Git 忽略 `output/`（原来的 `outputs/` 仍保留）。
- [ ] 新相机训练保存 Adapter、优化器、调度器、课程/步数和随机状态；不跨更新保存复用 KV。
- [ ] 明确新实验与 resume；完整接入训练、分布式及相机推理的运行目录管理。
- [ ] 训练保存 `training.txt` 和机器可读配置，禁止静默载入其他实验。

### 13. 测试与文档 — 路径/配置部分先行，其余待实现

- [ ] 几何、时间/空间对齐、首帧掩码测试。
- [ ] 目标块全采样链梯度、历史截断、冻结主干及仅 Adapter 更新测试。
- [ ] 零输出/关闭 Adapter 基线一致性、训练/推理采样一致性。
- [ ] 小样本过拟合和保存/恢复测试。
- [x] 路径/配置及原模型复现文档；CPU 验证结果见下方实施记录。

## 实施记录

### 2026-09-10：步骤 1、2；准备基础 LongLive 2.0 复现

- 创建以上清单和独立实验 A 规范；配置不冒充已实现训练。
- 接入 `utils/project_paths.py`，将模型组件路径贯通至训练/推理构造器。
- 新增 `configs/baseline/`、`run_baseline.py`、`utils/run_record.py` 与 `docs/baseline_reproduction.md`。
- 当前基础复现默认 32 latent / 125 RGB、4 步、BF16、无 Adapter/LoRA、无编译、VAE 与生成器同卡；S2/S4 独立匹配其权重。
- 验证：13 项 CPU 单元测试通过；仓库 19 份 YAML 配置解析/序列化重载通过；90 个 Python 文件语法检查通过；`git diff --check` 通过。
- 已从仓库外的 cwd 执行真实 launcher dry-run，正确解析 workspace/组件路径，保存完整配置与缺失输入报告。未启动 GPU 模型。
- 验证摘要：`output/verification_step01_02/verification.txt`；dry-run：`output/baseline_t2v_longlive2_bf16_s4_f32_seed0_20260910-120528-726132_step01-02-verification/`。
- CPU 测试入口：`python -m unittest discover -s tests -p test_project_paths.py -v`。本机临时配置依赖置于忽略目录 `output/verification_step01_02/deps`，没有修改项目依赖清单。
- 限制：当前检出目录没有用户的模型和数据；本机测试 Python 未安装 PyTorch，尚未进行 GPU 视频生成。CPU dry-run 不等同于复现成功。
- 下一步：用户验证原版 LongLive 2.0 后，再实施 CSV 接入；相机模型、损失和训练器保持待实现。

### 2026-09-10：明确 I2V 目标并纠正默认复现模式

- 用户明确项目 pipeline 为 I2V。此前默认 T2V 不符合该目标；已经生成的 `first-baseline` T2V 视频仅为基础文生视频对照，无对应 GT 或参考首帧。
- 将 launcher 默认配置改为 `configs/baseline/longlive_bf16_i2v.yaml`；新增 S4/S2 的 I2V 独立配置，并同步 README/复现说明。
- 当前 I2V 基线输入首帧和同名文本，沿用现有首帧 latent 固定逻辑；相机轨迹、CSV/GT 视频和控制训练仍未接入。
- 基础模型 I2V 复现与最终“首帧＋相机轨迹”的受控 I2V 分阶段验证，不能将前者视为已实现相机控制。
- 验证：13 项 CPU 测试通过，包含无 `--config` 时默认 I2V、首帧模式及 S4/S2 I2V 权重/步数匹配；21 份 YAML 解析/重载通过，`git diff --check` 通过。结果保存于 `output/verification_i2v_default/verification.txt`。GPU 推理未执行。

### 2026-09-10：步骤 3；RealEstate10K CSV / 相机 NPZ 数据接入

- 新增 `inspect_realcam.py`、`utils/realcam_dataset.py`、`utils/realcam_inspection.py`、`configs/data/realcam_inspection.yaml`、18 项数据测试及 `docs/realcam_csv.md`；同步 README、复现说明和实验 A 中已确认的 train CSV 路径。
- 按用户提供的 7 个 CSV 字段适配，保留长/短文本、align_factor、camera_scale、vtss_score。缺失内外参时按 video_path 关联官方 NPZ，不依赖行顺序；自动发现唯一匹配相机文件，也可覆盖路径。
- 官方整表 NPZ 每个 split 加载一次，限定 NumPy 对象类型；检查后的相机保存为独立数值缓存。记录原 CSV/NPZ 哈希、列映射、相机内容哈希，支持结果目录和数据根目录迁移。
- 用户说明上级目录为相似场景分组：默认有明确原视频 ID 时用原视频 ID，否则使用目录组，保留空 source_id 并显式记录原视频级泄漏检查未完成。依据 CSV 保留 train/test，按组划分 validation；抽查也扫描全表分组及重复路径。
- 对本地实际 `E:/codexspace/RealCam-Vid_test.npz` 完成相机检查：共 5,000 条，RealEstate10K 2,152 条、296,250 个相机帧，全部通过当前形状/有限值/齐次矩阵/旋转/内参/align_factor 检查；每条长度 16–279。原路径分布为 train 1,932 条、test 220 条，证明不能从目录重建 RealCam-Vid split。该检查不验证真实视频或时间对齐。
- 实际 NPZ 检查及全部参数：`output/verification_step03/real_npz_20260910-153118/inspection.txt`。
- 验证：18 项新数据测试＋13 项路径/基线测试，共 31 项通过，无跳过；覆盖真实合成 MP4 解码/损坏文件、CSV/NPZ 乱序关联、用户 CSV 字段、目录分组、跨集合重叠、相机缓存迁移与变化检测。22 份 YAML 在路径测试中解析/重载通过；164 个 Python 文件语法检查、`git diff --check` 通过。
- 从仓库外 cwd 运行真实 CLI，使用同样 7 列 CSV、官方格式 NPZ 和 125 帧 MP4 合成样本，正确生成 train/test 清单及完整参数记录。报告：`output/verification_step03/20260910-152850/verification.txt`；运行结果：`output/data_check_realestate10k_decode_step03_synthetic_20260910-152850/`。
- 尝试全仓库测试发现时，另有 7 个旧测试模块因本机缺少 torch/pytest 无法导入；不能宣称全部模型测试通过。本轮没有 GPU 推理/训练，也没有在真实 CSV 和视频上执行完整检查。
- 下一步：第 4 步实现 24 FPS / 125 帧连续窗口、窗口首帧提取、RGB/相机实际索引同步、裁剪缩放同步更新内参。当前 I2V 基线继续使用图片和同名文本，相机训练仍为 spec_only。
