# 已构建数据集的第三步统一验证

前提：RealEstate10K CSV/NPZ 已整理完成，第二步基础模型验证已通过。本流程不重新提取 NPZ、不运行独立 check-only，不需要旧 output 中的清单或缓存。正式推理仍会检查文件是否存在，但不另建检查目录。

数据应保存在 workspace/public_data/RealCam-Vid/RealEstate10K：

```text
RealEstate10K/
  RealEstate10K_train.csv
  RealEstate10K_train.npz
  RealEstate10K_test.csv
  RealEstate10K_test.npz
  train/<目录>/<视频>.mp4
  test/<目录>/<视频>.mp4
```

CSV 使用 dataset_source，video_path 相对该数据根目录。四份 CSV/NPZ 必须位于数据目录；本入口不会回退到原 RealCam-Vid 归档，也不会查找已删除的 output。

## 只运行两条命令

在仓库目录执行：

```bash
python validate_realestate10k.py --run-name realestate10k_v1
```

这一步不加载模型。它依次检查 CSV/相机匹配、视频解码、集合分组，再选择可用的连续窗口，并提前把对应 GT 和首帧放入 comparisons。默认每个 split 检查前 50 条（分组仍扫描全表），从 test 清单按顺序选取 10 个不同 clip，每个 clip 一个窗口。不是随机场景抽样，也不代表全量数据质量已通过验证；上级目录仍只是相似场景组，不能保证原视频级来源独立。

默认窗口为 125 帧、24 FPS、704×1280；从所选 5B I2V baseline 配置读取 latent 帧数和尺寸，保持输入与生成尺寸/时长一致。相机数组同步检查，但尚未作为模型输入。

成功时 summary.json 的 status 为 prepared、exported 为 10，终端和 next_command.txt 只给出：

```bash
python validate_realestate10k.py --infer output/realestate10k_v1
```

这一步在同一目录生成 10 个视频，使用保存的配置、seed 和输入快照。默认 BF16 四步 I2V，num_samples=1、inference_iter=-1，每张首帧生成一次。模型推理参数与原基线一致，不引入相机 Adapter。生成数量正确且推理返回成功才标记 completed；completed 表示流程完成，视觉运动质量仍需对照验收。

## 只需关注这些文件

```text
output/realestate10k_v1/
  validation.txt             # 结果摘要和查看说明
  summary.json               # 状态、数量、拒绝原因
  next_command.txt           # 准备成功后的推理命令
  comparisons/
    0000/
      input.png              # 模型输入首帧
      ground_truth.mp4       # 同一窗口的真实视频；准备完成后即存在
      generated.mp4          # 生成视频；推理完成后存在
      prompt.txt             # 实际分块提示词；推理后保存
      sample.json            # 视频来源、帧索引、裁剪等窗口参数
      reference.json         # 输入/GT 对应关系与哈希
    0001/ ... 0009/
  records/
    parameters.json          # 本次全部入口参数
    baseline_config.yaml     # 保存的基线配置
    input_snapshot.json      # 输入图片和文本的内容快照
    next_command.json        # 下一条命令的结构化记录
    inspection/              # CSV/相机清单、缓存、配置、拒绝记录
    windows/                 # 窗口索引、完整导出、配置、拒绝记录
    inference/               # 有效模型配置、环境、console.log、运行状态
```

GT 为裁剪、重采样后同一段窗口的有损视频预览，不是整个原始视频，也不参与推理。直接比较相同编号中的两个 mp4，记录运动停止的时间、是否完全冻结、GT 是否持续运动。模型当前只接收首帧与文本，不保证复现 GT 的相机轨迹。

records 保留复现和后续训练接口需要的数据，不必逐个打开。检查失败时，先看顶层 summary.json 的 error，再按提示查看对应 records 子目录中的拒绝记录。

## 调整样本数或重跑

不足 10 个可用片段时仍保留已找到的 GT，并标记 preparation_failed，不重复凑数、不提示推理。扩大候选数量并使用新的名称：

```bash
python validate_realestate10k.py --run-name realestate10k_v2 --count 10 --limit 200
```

可用 --split train 选择训练集，--seed 1 改变确定性的窗口/裁剪与生成随机种子，--baseline-config configs/baseline/longlive_nvfp4_s2_i2v.yaml 选择另一已验证的配置。--limit 0 检查全表；--count 决定最终导出数量。

现有运行目录不覆盖；准备或推理失败后，新一轮验证使用新 run-name。旧输入不得在准备后被替换。路径均通过仓库/workspace 相对位置解析；非默认位置可传 --workspace-root 与 --data-root，无需 export 环境变量。迁移平台后应从第一条命令重新准备，避免复用旧机器的解析路径。

原 inspect_realcam.py、prepare_realcam_windows.py、run_baseline.py 作为底层与专项工具保留；extract_realestate10k.py 仅在以后需要重建数据时使用，不属于本次主流程。临时 diagnose_realcam_temp.py 和旧结果 collect_baseline_references.py 已删除。
