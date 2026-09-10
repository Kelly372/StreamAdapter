# StreamAdapter 代码修改与实施记录

本文件是本项目的持续维护清单。后续每次实施应同步更新任务状态、修改文件、验证结果、限制和下一步；只有配置定义完成时，不将对应训练功能标记为已实现。

## 全局约定

- 目标目录：`workspace/project/StreamAdapter`。当前本地检出的目录名可以不同。
- 数据：`workspace/public_data/RealCam-Vid` 中的 RealEstate10K；元数据为 train/test CSV。
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
- [x] 提供原版 BF16 T2V/I2V、NVFP4 S4/S2 复现配置与 `run_baseline.py`。
- [x] 复现入口启动前检查文件，支持无 GPU 的 dry-run/check-only。
- [x] 复现结果存入 `output/<run_name>/`，同名运行拒绝覆盖，并保存参数/状态。
- [ ] 在用户实际模型和 CUDA 环境执行基础模型推理，确认视频输出。
- 当前统一记录入口为 `run_baseline.py`（单进程单卡）；原始 train/inference_sp 的完整运行目录生命周期将在第 12 步接入。

### 3. RealCam-Vid CSV 数据接入与检查 — 待实现

- [ ] 核查 CSV 列名、视频/相机/文本路径及来源标识，读取 train/test 并筛选 RealEstate10K。
- [ ] 检查文件、FPS、帧数、位姿长度/有效性，保存剔除原因与统计。
- [ ] 保留 train/test 划分；训练期间验证集从 train 按原始视频来源划分。
- [ ] 保留原始视频，以窗口索引采样，不强制转换为官方目录格式。
- 建议：`utils/realcam_dataset.py`、数据检查脚本。

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
