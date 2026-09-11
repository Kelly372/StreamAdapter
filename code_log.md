# StreamAdapter 代码修改与实施记录

本文件是本项目的持续维护清单。后续每次实施应同步更新任务状态、修改文件、验证结果、限制和下一步；只有配置定义完成时，不将对应训练功能标记为已实现。

## 全局约定

- **项目目标固定为 I2V：首帧图像＋文本＋相机轨迹 → 受控视频。基础复现也默认 I2V；T2V 仅作可选对照。**

- 目标目录：`workspace/project/StreamAdapter`。当前本地检出的目录名可以不同。
- 数据根目录（2026-09-11 更新）：`workspace/public_data/RealCam-Vid/RealEstate10K`；原始总 NPZ 保留在其父目录 `RealCam-Vid`。
- 用户手动整理 CSV 并将 video_path 改为 `train/<目录>/<视频>.mp4`、`test/<目录>/<视频>.mp4`；CSV 和新子集 NPZ 放在上述数据根目录。字段为 `dataset_source, video_path, short_caption, long_caption, align_factor, camera_scale, vtss_score`。CSV 和官方 NPZ 均为 dataset_source=RealEstate10K，data_source 仅兼容。第 3A 步以 CSV 集合归属及顺序为准，从两份原始 NPZ 查询并重建对应 NPZ；不依据归档名或视频路径目录重新划分 CSV。上级视频目录仍只作为相似场景分组，不标记为原始视频 ID。
- Wan2.2 基础组件：`workspace/pretrained_model/Wan2.2-TI2V-5B`。
- LongLive 蒸馏权重：`workspace/pretrained_model/LongLive` 下的 `LongLive-2.0-5B`、`LongLive-2.0-5B-NVFPS-S4`、`LongLive-2.0-5B-NVFPS-S2`。
- `NVFPS` 是用户提供的本地目录拼写，数值格式仍为 NVFP4；实际目录若为 `NVFP4`，使用配置/命令行覆盖。
- 相对目录结构保持稳定，绝对路径允许变化；使用配置和命令行，尽量不要求手动 export。
- 结果放在仓库 `output/<有区分性的运行名称>/`；保存最终生效参数（含默认值）、启动信息、代码版本和数据/权重路径。训练记录为 `training.txt`，推理为 `inference.txt`。
- 原始相对配置与本次解析的绝对路径分别保存，便于迁移和审计。
- 数据目录：
    Workspace/public_data
        - RealCam-Vid
            - RealCam-Vid_train.npz
            - RealCam-Vid_test.npz
            - RealEstate10K
                - RealEstate10K_train.csv
                - RealEstate10K_test.csv
                - RealEstate10K_train.npz
                - RealEstate10K_test.npz
                - test
                    - sub_dir/video_name.mp4
                - train
                    - sub_dir/video_name.mp4
- csv文件示例：
dataset_source,video_path,short_caption,long_caption,align_factor,camera_scale,vtss_score
RealEstate10K,train/ZPLUfZsgEtg/f3fa5c1e24a522bc.mp4,"...","...",3.67878591096826,1.5096761946889783,0.06628306

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

### 3. RealCam-Vid CSV 数据接入与检查 — 已完成（用户确认）

- [x] 接入用户确认的 CSV 列名/相对视频路径，读取 train/test 并筛选 RealEstate10K；支持列名映射。
- [x] 按 video_path 关联官方相机 NPZ；另支持列内数组、JSON/数值 NPZ/内外参 NPY 引用。保留相机坐标及 align_factor，不提前做几何转换。
- [x] 检查文件、FPS、实际解码帧数、位姿长度/有效性、可用时间戳，保存剔除原因与统计；明确区分 metadata/header/decode。
- [x] 保留 CSV 原始 train/test；验证集从 train 按明确原视频 ID 或场景目录稳定分组，分别记录分组依据。检查重复路径及组间重叠，抽查时也扫描完整分组；无真实来源 ID 时不宣称完成原视频级泄漏检查。
- [x] 保留原始视频，提供可迁移清单和按需相机读取接口；实际窗口索引与视频张量采样在第 4 步实现。
- [x] 独立 output 子目录记录全部配置、inspection.txt、数据哈希/字段映射、运行环境、清单和状态；拒绝同名覆盖。
- [x] 已读取本地实际 `E:/codexspace/RealCam-Vid_test.npz`，核实字段、数组形状及部分 test 条目指向原始 train 视频目录；不能从路径推断 RealCam-Vid 的集合划分。
- [x] 用户确认第 3 阶段已完成，可进入第 4 阶段；服务器真实数据运行指标以用户报告为准。本机未额外取得该报告，不新增真实性能/数量结论。原视频 ID 仍未知，当前按目录组检查，后续有可靠 ID 时可进一步核实来源级泄漏。
- 文件：`inspect_realcam.py`、`utils/realcam_dataset.py`、`utils/realcam_inspection.py`、`configs/data/realcam_inspection.yaml`、`docs/realcam_csv.md`。

#### 3A. 预先分离 RealEstate10K 相机 NPZ — 已实现

- [x] 2026-09-11 改为默认按 CSV 划分：从两份原始 NPZ 精确查询，生成短路径的 train/test 子集 NPZ；检查 CSV 内/跨集合重复、缺失和元数据冲突，记录来源并往返核验。下列早期“保持原归档划分”的行为仅在显式 --split-policy archive 时保留。

- [x] 新增 `--inspect-only` 原始 NPZ 查看模式，默认仅查看 test；不筛选子集，导出完整 JSONL、精简路径索引、展开预览与字段/来源统计，供人工核对。

- [x] 新增 `extract_realestate10k.py`，从 `RealCam-Vid_train/test.npz` 按 dataset_source 分离 `RealEstate10K_train/test.npz`，支持分别指定源文件或只处理一个 split。
- [x] 保持源 NPZ 的 train/test 归属和条目顺序，不依据视频目录名重新划分，不改变任何字段或相机数组。
- [x] 输出保存于仓库 `output/<run_name>/`，含 extraction.txt、全部参数、环境、输入输出哈希、数量统计、状态；不覆盖输入和已有结果。
- [x] 输出重新读取后逐字段校验类型/值及数组 dtype/形状/内容。缺失 dataset_source、零匹配或校验失败明确报错。
- [x] 检查器支持 `--camera-metadata-dir` 直接关联分离结果，原 CSV/视频不搬动；自动发现 NPZ 时子集表优先于总表。
- [x] 修复 CSV 总表/子集表并存的选择：优先唯一 RealEstate10K 表，兼容已报告的 RealState10K 文件名；多个同级候选仍要求显式路径。
- 分支为可选数据准备步骤，不能替代视频解码检查，不推进第 4 步窗口采样。

### 4. 连续窗口采样与几何同步 — 已实现，待服务器真实数据验收

- [x] 基于视频解码 PTS，以 24 FPS 目标时间间隔采样 125 帧；RGB/相机使用相同实际索引，严格禁止重复，记录实际与目标时间误差。
- [x] 首帧取当前窗口首帧；全窗口共享覆盖缩放＋随机/中心裁剪，内参按真实缩放尺寸及 half-pixel 像素中心映射同步更新，输出像素 K。
- [x] 不启用未同步的翻转/旋转/逐帧裁剪；避开无效位姿、时间空洞、已标注/检测到的切镜，不跨这些边界拼接，不复制/循环补帧。
- [x] 提供自动切镜检测和显式标注两种模式。自动检测可能漏检/误检，不能视为所有真实切镜均已验证；结果记录边界来源，待人工检查或可靠标注补充。
- [x] 返回原来源/目录组、帧索引、实际/目标/相机时间戳、有效掩码、空间变换和相机约定；外参/align_factor 不提前转换。
- [x] 建立可迁移窗口索引，按 seed/sample/epoch/draw 可复现地选取窗口；原清单、相机、视频及时间缓存变化时校验失败。
- [x] 独立 output 目录导出首帧、GT 预览、camera.npz、sample.json、baseline_i2v/、sampling.txt 和完整参数；原基线可以直接读取导出首帧/文本。
- [ ] 在服务器真实视频上抽查窗口、切镜与几何对齐；确认可用率后建立完整 train/validation/test 窗口索引。
- 文件：`utils/realcam_windows.py`、`prepare_realcam_windows.py`、`configs/data/realcam_windows.yaml`、`tests/test_realcam_windows.py`、`docs/realcam_windows.md`。Torch 训练器接入仍属于第 7 步。

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

- [x] 第 4 步 RGB/相机时间索引、缩放裁剪内参、条件首帧一致性测试。
- [ ] 第 5 步射线几何及第 10 步 latent 首帧监督掩码测试。
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

### 2026-09-10：步骤 3A；分离 RealEstate10K NPZ 并修复文件选择歧义

- 新增 `extract_realestate10k.py` 及 `tests/test_realcam_extraction.py`。默认依次读取总 train/test NPZ，只筛选 dataset_source，保存同格式的 arr_0 字典数组；保留所有字段、原划分、相机数组 dtype/形状/内容及原顺序，输出后逐字段重读校验。
- 默认结果放在 `output/metadata_realestate10k_<splits>_<时间>_<tag>/`，也可用 `--run-name` 指定唯一子目录；记录 extraction.txt、parameters.json、launch.json、summary.json、status.json 和输入输出 SHA256。不自动改写数据目录。
- `inspect_realcam.py` 新增 `--camera-metadata-dir`（仓库相对路径），直接使用分离结果。CSV 自动选择优先唯一子集表，兼容用户报告的 RealState10K 拼写；两种子集拼写同时存在仍要求显式选择。相机 NPZ 自动选择优先 RealEstate10K，再回退 RealCam-Vid；显式路径或目录覆盖优先。
- 已实际运行本地 test 分离：5,000 → 2,152 条，排除 2,848 条，输出 27,860,909 字节，全字段往返校验通过。结果及参数：`output/metadata_realestate10k_test_20260910-173628-455784_step03a-local-verification/`。本机未提供总 train NPZ，真实 train 分离尚未执行；train/test 双文件流程已用合成数据验证。
- 验证：新增 5 项测试，连同已有 31 项，共 36 项通过，无跳过；覆盖原始顺序/dtype/嵌套字段保留、原文件不变、同名拒绝覆盖、空子集/缺失字段失败、CSV 同级歧义、分离结果到检查器的双 split 串联、单 split 和缺失输入记录。166 个 Python 文件语法检查、`git diff --check` 通过。
- 验证记录：`output/verification_step03a/20260910-173931/verification.txt` 和同目录 parameters.json；同步 README、`docs/realcam_csv.md` 与数据检查默认配置。未执行 GPU 推理或训练；第 4 步保持待实现。

### 2026-09-10：步骤 4；连续窗口、首帧与相机几何同步

- 用户确认第 3 阶段完成后，新增 `utils/realcam_windows.py`、`prepare_realcam_windows.py`、`configs/data/realcam_windows.yaml`、`tests/test_realcam_windows.py`、`docs/realcam_windows.md`；同步 README、复现/CSV 说明、实验 A 数据接口约定和环境中的 Pillow 版本记录。
- 从第 3 步清单扫描真实视频 PTS，按有效位姿、时间间隔和检测/标注切镜构造连续区间，枚举可行源帧起点；125 个 24 FPS 目标时间各对应一个唯一源帧，最大时间误差默认 1/48 秒。无有效窗口的短片、低 FPS 或不连续片段明确剔除，无重复、补帧或跨镜头拼接。
- 引入 RGB/颜色直方图切镜检测和严格逐视频标注模式。检测阈值、缩略图尺寸、直方图 bins 记录在配置中；自动检测可能漏检/误检，索引与样本显式记录边界来源，服务器样本仍需人工核查。
- 同一窗口共享一个覆盖缩放和随机/中心裁剪；按实际 sx/sy 与 half-pixel 映射更新 K，统一像素内参和整数像素中心，避免未来第 5 步射线网格约定不一致。相机外参、align_factor 保持原约定和数值，本阶段不做窗口参考系变换。
- `RealCamWindowDataset` 提供 float32 NumPy [125,3,704,1280] 的 video、严格等于 video[0] 的 image、文本、相机、有效掩码、实际/目标时间戳和来源信息；按 seed/sample/epoch/draw 可复现，读取顺序不影响结果。训练器及多进程持久 worker 的 epoch 同步留在第 7 步。
- 输出包含窗口索引/时间缓存、首帧 PNG、GT MP4 预览、camera.npz、sample.json、baseline_i2v/、sampling.txt 与全部配置/环境/状态。多行 caption 在基线文本中折叠为单行，原文保留，避免原 I2V 入口误解释为多个镜头。GT MP4 为有损预览，不作为像素级缓存。
- 验证：新增 14 项 CPU 测试，连同既有 36 项共 50 项通过，无跳过；覆盖 24/30 FPS、真实 VFR、近邻唯一索引、时间空洞/无效位姿/切镜、内参投影和像素中心、首帧一致性、不同 epoch、文件变化及跨路径迁移、导出/失败记录。169 个 Python 文件语法检查、23 份 YAML 解析/重载和 `git diff --check` 通过。
- 从仓库外 cwd 执行完整默认分辨率的合成样本流程：210 帧 30 FPS 源视频，选中起点 4，导出 125 帧、24 FPS、1280×704 GT；最大时间误差 0.016667 秒，相机平移中的帧编号与 RGB 索引一致，PNG 尺寸和首帧一致性验证通过。
- 验证及全部参数：`output/verification_step04/20260910-184647/`（tests.txt、parameters.json、verification.txt、artifact_check.json）；实际导出：`output/windows_realestate10k_step04_fullres_synthetic_20260910-184647/`。
- 限制：本机仍无服务器真实视频，未进行真实窗口可用率/切镜质量验收，也未运行 GPU 或旧模型测试。默认窗口约 1.26 GiB CPU 内存，建全量索引可用 --export-count 0。下一步先服务器少量窗口核查，再进入第 5 步相机参考系变换与 Plücker 编码。

### 2026-09-10：按最新目录与七字段 CSV 核对、修正读取流程

- 初次依据示例将 CSV 的 `data_source` 设为首选；用户随后纠正实际字段为 `dataset_source`，此项已撤回。当前 CSV 与官方 NPZ 均以 `dataset_source` 为准，`data_source` 仅为兼容别名，仍支持显式 `data.columns` 映射。
- CSV 自动发现优先数据根目录直接包含的 RealEstate10K train/test 表，避免递归搜索到备份/导出同名表导致歧义；根目录没有子集表时才沿用递归查找，同优先级多个候选仍要求指定。显式 CSV 配置优先。
- 视频路径继续仅相对 `workspace/public_data/RealCam-Vid` 拼接，不重复追加 RealEstate10K 或 train/test；兼容 Windows/Linux 分隔符。CSV 提供文本和非空 align_factor，NPZ 提供内外参；完整 video_path 精确关联，集合归属依据 CSV，不猜测或改写路径。
- 检查过程显示实际 CSV；`metadata_sources.json` 记录视频路径基准和相机匹配规则；`missing_camera_metadata` 增加 CSV、所选 NPZ、实际视频路径，便于区分选错文件与标注缺失。该改动不能补齐 NPZ 中不存在的轨迹，未声称服务器此前 10 条拒绝已解决。
- 修改：`utils/realcam_dataset.py`、`utils/realcam_inspection.py`、`configs/data/realcam_inspection.yaml`、`tests/test_realcam_dataset.py`、`tests/test_realcam_windows.py`、`docs/realcam_csv.md`。
- 验证：52 项 CPU 测试通过、无跳过，169 个 Python 文件语法检查和 `git diff --check` 通过。新增根目录/备份 CSV 选择验证，以及七字段 CSV → 官方对象 NPZ → 真正 MP4 解码 → 迁移后的窗口加载验证；覆盖 CSV/NPZ 不同来源字段名、路径分隔符、CSV 文本与 align_factor 优先、RGB/相机索引及首帧一致性。
- 测试使用合成视频，未验证服务器真实数据或 GPU。全部验证参数、环境、配置和结果：`output/verification_data_layout/20260910-201753-698734/`（parameters.json、tests.txt、verification.txt）。
- 使用更新后的代码重新执行 `python inspect_realcam.py --limit 10 --tag layout-v2`；若 NPZ 在第 3A 步输出目录，继续显式传入 `--camera-metadata-dir output/<分离结果目录>`。第 4 步的 `--manifest` 指向本次新生成的检查清单；修改过输入表/NPZ 时不能复用旧窗口索引。

### 2026-09-10：更正 CSV 来源字段为 dataset_source

- 按用户最新纠正，CSV 和官方 NPZ 的正确字段均为 `dataset_source`。恢复该字段为自动映射首选，`data_source` 仅作兼容；同步配置注释、本文全局约定、CSV 文档及七字段窗口验证。用户已修正的 CSV 示例保持原样。
- 验证：2 项相关集成测试通过，覆盖 CSV/NPZ 关联、缓存与视频窗口迁移、RGB/相机索引及首帧；另核对双字段存在时优先 dataset_source、仅有 data_source 时仍兼容。`git diff --check` 通过。
- 参数及验证记录：`output/verification_dataset_source_20260910-202758/`。本次不涉及服务器真实数据或 GPU 验证。

### 2026-09-10：成功后打印下一步命令

- 第 3 步成功时在原摘要后打印窗口准备命令，引用本次 train.csv，默认抽查 10 条、导出 2 条；例如 audit_smoke_v2 → windows_smoke_v2。保留非默认 workspace、数据根目录、seed、目标帧数/FPS。退出码非 0 时不打印继续命令。
- 第 4 步导出成功后打印 BF16 I2V 的 check-only 命令，输入指向本次 baseline_i2v；仅建索引时先建议用原配置/覆盖参数导出首帧和文本，不引用尚不存在的图片目录。
- baseline check-only 成功后打印正式推理命令，保留配置、workspace、所有覆盖参数和 tag；dry-run 输入齐全时先建议 check-only，缺失输入时不建议继续。正式推理是当前流程终点，不生成未实现阶段的命令。
- 下一步结果使用新名称，已有目录追加编号；命令适用于当前平台的 Bash/POSIX shell 或 PowerShell，对空格和特殊字符正确引用。在仓库目录复制执行，不自动运行；每次建议同时写入 next_command.txt 和 next_command.json（argv、cwd、shell）。跨阶段图片生成参数由基线配置控制，使用非默认窗口尺寸/时长时仍需核对。
- 修改：`utils/next_command.py`、`inspect_realcam.py`、`prepare_realcam_windows.py`、`run_baseline.py`、`tests/test_next_command.py` 及相关使用文档。
- 验证：56 项 CPU 测试通过、无跳过；171 个 Python 文件语法检查、git diff --check 通过。新增测试覆盖第三步成功/失败、仅索引→执行建议导出→基线检查、推理参数和含空格路径保留、失败不提示、重名避让。GPU 推理未运行。
- 参数及结果：`output/verification_next_command_20260910-204150/`（parameters.json、tests.txt、verification.txt）。

### 2026-09-10：临时诊断第三步的 CSV/NPZ 匹配失败

- 新增独立 `diagnose_realcam_temp.py`。可读取 `--audit-dir` 中实际配置、NPZ 选择及 rejected.csv，优先检查失败路径；无历史结果也可传 `--camera-metadata-dir`。默认每个 split 最多 10 条，支持 `--splits test` 聚焦当前失败集合。
- 复用生产 CSV 字段映射、子集识别、路径规范化和受限 NPZ 读取器；记录 CSV/NPZ 哈希、原始/规范化路径、字段映射、文件存在性和各相机归档的精确匹配。失败时顺序对照可用总表和另一 split 的 NPZ，提供大小写、相同场景/文件名等线索，不自动改写路径或切换生产输入。
- 每次使用时间戳新建 output/diagnose_realcam_temp_*，避免复用已存在审计目录；保存 diagnosis.txt、report.json、parameters.json、resolved_config.yaml、status.json，以及可用的旧报告摘要。退出码 0 仅表示抽查路径匹配/文件存在/文本非空，1 表示发现问题，2 表示诊断无法完成。
- 不解码视频、不校验相机几何、不加载模型，也不修改 CSV/NPZ/原视频；大型 NPZ 仍需完整解压一次，按文件顺序释放。当前不能宣称服务器 test=0 已解决。
- 使用：`python diagnose_realcam_temp.py --audit-dir output/realestate10k_audit_smoke_v2 --splits test --limit 10`。如输出目录已移动或配置需调整，传 --workspace-root / --set 覆盖；若显式指定新相机目录则检查该新目录。
- 验证：4 项临时诊断测试通过，2 个新增 Python 文件语法检查通过；覆盖正常匹配、重复运行/原文件不变、所选子集缺条目但总表精确匹配、路径前缀不同、旧拒绝路径优先、旧实际 NPZ 选择、缺失输入的失败记录。参数及测试记录：`output/verification_diagnose_temp_20260910-205932/`。
- 脚本为临时分支工具；服务器定位并完成第三步验证后，可删除该脚本、`tests/test_diagnose_realcam_temp.py` 及 docs/realcam_csv.md 的临时说明，保留本日志作为追踪记录。生产加载器不依赖临时脚本。

### 2026-09-11：第 3A 步增加原始 NPZ 可读导出

- `extract_realestate10k.py --inspect-only` 默认只读取 `RealCam-Vid_test.npz`；可用 --test-npz、--splits、--workspace-root、--data-root 覆盖。未启用该模式时，原双 split 子集提取行为不变。
- 新增 `utils/realcam_readable.py`，复用受限 NPZ 读取器，将全表所有字段及数组数值按原顺序写入 records.jsonl；index.jsonl 仅保留序号、来源、路径和数组形状/dtype，preview.json 展开前若干条完整记录。--preview-count 默认 3，只控制终端摘要和预览数量，不截断完整导出。
- summary.txt/json 汇总总数量、dataset_source、路径前缀、字段类型及数组形状/dtype；终端仅打印前 10 种主要路径前缀，全部计数保存在文件中。数组包含 dtype、shape、values；NaN/Infinity 使用显式标记对象，元组转为列表。该 JSON 格式供人工检查，不作为替代 NPZ 的训练输入。
- 输出使用独立 output/metadata_readable_* 目录，含全部参数、extraction.txt、环境、状态及输入 SHA256；导出前后校验原 NPZ 哈希不变。不筛选 RealEstate10K、不改写路径或划分、不读取视频/CSV，也不推进下一阶段检查。
- 已对本地真实 `E:/codexspace/RealCam-Vid_test.npz` 执行：5,000 条全部导出，RealEstate10K 2,152、MiraData9K 1,895、DL3DV-10K 953。RealEstate10K 路径前缀 train 1,932 / test 220；该统计只描述路径，不能视为归档划分。
- 实际结果及参数：`output/metadata_readable_test_20260911-092713-881116_local-readable/`，完整 records.jsonl 为 152,688,787 字节。验证两个 JSONL 均为 5,000 行、预览 3 条，首条外参形状为 [92,4,4]。
- 6 项相关测试通过、无跳过，覆盖原提取流程、只读模式默认 test、全子集/全字段保留、数组 dtype/形状/数值、NaN 标记、预览数量及原文件不变；3 个相关 Python 文件语法检查通过。验证记录为上述目录 verification.txt、verification_tests.txt。GPU 和服务器真实 CSV/视频未测试。

### 2026-09-11：迁移至 RealEstate10K 根目录并按 CSV 重建 NPZ

- 主体默认数据根目录改为 public_data/RealCam-Vid/RealEstate10K，涵盖第三步、第四步、BF16/NVFP4 基线、实验 A 与路径默认值。分组规则接受 train/<场景>/<视频>、test/<场景>/<视频> 和 ./ 前缀，显式旧根目录仍兼容旧的 RealEstate10K/... 路径。下一步命令按新默认根目录传递配置。
- 第 3A 步默认 --split-policy csv；--source-root 默认 public_data/RealCam-Vid，原始 NPZ 相对该目录；--data-root 默认其 RealEstate10K 子目录，CSV 相对该目录。两份源 NPZ 均用于查找所需记录，生成集合和顺序由 CSV 决定，解决已观察到的 test CSV 相机位于原 train NPZ 的问题。--inspect-only 继续只读原始 test NPZ；旧分离方式保留为显式 --split-policy archive。
- 不修改用户 CSV。原 NPZ 路径只去掉 RealEstate10K/ 前缀并规范化分隔符，保留 train/test 目录名；输出所有其他字段及相机数组 dtype/形状/数值不变，包括可变序列长度。按同一个规范化键检查 CSV 重复及跨集合重复；两份源归档的同键记录只有全部字段一致时才合并，任何差异均报告冲突。
- 缺失/冲突会写 partition_report.json 并失败，生成子集 NPZ 前完成这些检查；成功后重新读取，核验路径集合/顺序与 CSV 一致、全部字段往返一致。train/test_provenance.json 保存每条 CSV 行的来源归档，报告中记录全部 CSV/NPZ 哈希、参数和输入/输出统计。
- 源归档逐份加载，命中的记录暂存于输出目录内 SQLite，结束时删除单个暂存文件；每份输出分别组装和校验，避免同时保留两份完整源数组。仍需足够内存与磁盘空间。未自动移动任何用户文件，结果仍在唯一 output 子目录。
- 新流程：用户先整理新目录的 CSV → `python extract_realestate10k.py --run-name realestate10k_csv_aligned_v1` → 使用成功后打印的第三步命令；或用户将生成 NPZ 移入 RealEstate10K 目录，再运行 `python inspect_realcam.py --limit 10 --tag relocated-smoke`。旧 CSV 哈希和窗口索引不复用，需要重建第三、四步结果。
- 验证：65 项 CPU 测试通过、无跳过，176 个 Python 文件语法检查通过。新增端到端验证使用来自两份原始 NPZ 的 test CSV 样本、混合 train/test 视频目录、不同相机帧数；生成文件移动后按新默认目录解码真实合成 MP4，第三步接受 train=1/test=2，第四步输出 125 帧且首帧/RGB/相机索引一致。另覆盖缺失、冲突、相同重复归并、CSV 跨集合重复、路径越界与前缀规范化。旧布局测试通过显式根目录覆盖验证兼容。
- 参数及验证结果：`output/verification_csv_partition_20260911-095720/`。本机尚无用户修改后的真实 CSV 和完整 train NPZ，因此服务器真实划分与第三步验收仍由用户运行；本轮未做 GPU 验证。
