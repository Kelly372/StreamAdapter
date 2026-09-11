# 第 4 步：连续窗口采样与几何同步

这一阶段将第 3 步的样本清单变成可读取的 I2V 窗口：125 帧 GT、同窗口首帧、文本和逐帧相机参数。默认 24 FPS、704×1280，对应实验 A 的 32 latent 帧和 44×80 latent 空间网格。当前只实现 CPU 数据处理；不加载 LongLive、不训练 Adapter。

## 服务器运行

输入必须是**第 3 步输出的** train/validation/test 清单，旁边应有 metadata_sources.json，以及该次检查产生的 cameras/ 缓存。不能把原始 RealEstate10K CSV 当作已检查清单传入。

```bash
# 将 <audit_run> 替换为实际的第 3 步结果目录名
python prepare_realcam_windows.py --manifest output/<audit_run>/train.csv --limit 50 --export-count 10 --tag baseline10

# 全量建立训练窗口索引，不导出预览视频
python prepare_realcam_windows.py --manifest output/<audit_run>/train.csv --export-count 0 --tag train-window-index

# 验证集采用中心裁剪，单独保存索引
python prepare_realcam_windows.py --manifest output/<audit_run>/validation.csv --set windows.crop_mode=center --export-count 2 --tag val-window-check
```

`--manifest` 和 `--shot-annotations` 相对仓库根目录，`paths.data_root` 相对 workspace，默认 `public_data/RealCam-Vid/RealEstate10K`；支持 `--workspace-root`、重复 `--set KEY=VALUE`。无需手动 export 环境变量。默认配置为 `configs/data/realcam_windows.yaml`。CSV 改为短路径后应使用新第三步清单；旧结果绑定原始 CSV 哈希。

默认导出 10 个不同可用 clip，每个 clip 一个窗口，按清单顺序选择；不是同一张首帧生成 10 次，也不是随机抽取 10 个场景。`--limit 50` 检查清单前 50 条作为候选，最终可能不足 10 个可用窗口；summary.json 中 requested_exports/exported_samples/export_shortfall 和终端提示会明确数量，不重复视频凑数。若第三步清单本身不足 10 条，先用 `python inspect_realcam.py --limit 50 --tag baseline10` 扩大检查（保留原来的数据/NPZ覆盖参数），然后使用新清单；候选仍不足时再提高限制。输入应已在第 3 步完成集合和目录组重叠检查，第 4 步不重新划分集合。

## 时间与连续性规则

1. 用 PyAV 完整解码每个待检查视频，保留真实帧时间戳，不依靠平均 FPS 推算全部帧位置。相机长度必须与解码帧数相同；若有相机时间戳，去掉各自起始偏移后逐帧核对。
2. 将视频按有效位姿、切镜边界及过大的时间间隔划成连续区间。默认相邻源帧间隔超过 0.0625 秒便切开，不跨缺失位姿或大时间空洞。
3. 从区间中的某个源帧开始，建立 `t_start + k/24`，k=0…124 的目标时间网格，选择区间内最近的真实帧，等距时选前一帧。
4. 所选源帧索引必须严格递增，时间误差不超过 1/48 秒。不复制、不循环、不插值 RGB 或相机；低帧率、短片或无法对齐的片段以 `no_valid_window` 剔除。
5. RGB 和相机使用完全相同的实际索引。窗口首帧就是 I2V 条件图像。实际采样时间可能与理想 24 FPS 网格略有差异，二者分别保存。

建立索引时枚举所有可行的源帧起点。读取时按 seed、sample_id、epoch、draw 生成独立随机数，从当前片段的可行起点中均匀选一个。同样参数可复现，不依赖读取顺序；不同 epoch/draw 可改变窗口和裁剪。此处是一条可用视频对应一个 dataset 条目，并非将所有可行窗口平铺成独立训练样本。

## 切镜约束

默认 `windows.shot_mode=detect`，使用缩略图 RGB 平均差异和颜色直方图差异检测突变，参数全部在配置中。算法会排除检测到的边界，但**可能漏检切镜，也可能误判快速运动或闪光**。因此当前不声称所有真实切镜都已被识别；需人工检查样本，或提供可靠的边界标注。

标注 JSON 以数据根目录下的 video_path 为键，值为“新镜头第一帧”的 0 起始索引。例如 150 表示第 149 与 150 帧之间切镜；空列表表示标注确认整段无切镜：

```json
{
  "train/group_a/clip_a.mp4": [150, 300],
  "train/group_b/clip_b.mp4": []
}
```

```bash
python prepare_realcam_windows.py --manifest output/<audit_run>/train.csv --shot-annotations annotations/cuts.json --set windows.shot_mode=annotated --tag annotated-windows
```

`annotated` 模式要求每条待处理视频都有标注，缺失时剔除。`detect` 模式也可传入标注，将标注和检测边界合并。summary.json 记录未匹配清单的标注键，避免路径拼写错误被忽略。两种模式均不跨已知边界拼接视频。

## 缩放、裁剪与内参

默认等比缩放覆盖目标尺寸（`resize_mode=cover`），再对整个窗口使用**同一个**随机裁剪框（`crop_mode=random`）。可选择中心裁剪或直接拉伸；没有水平翻转、旋转、逐帧随机裁剪等增强。

原归一化内参先转换到原视频像素单位，然后乘以实际图像变换矩阵。输出 K 统一为 `[F,3,3]` 像素内参。使用整数像素中心坐标：左上像素中心 `(0,0)`。Pillow 双线性 resize 的中心映射为：

```text
sx = resized_width / source_width
sy = resized_height / source_height
u_out = (u_in + 0.5) * sx - 0.5 - crop_left
v_out = (v_in + 0.5) * sy - 0.5 - crop_top
K_out = A @ K_in_pixels
```

缩放后的宽高可能向上取整，sx/sy 使用实际尺寸分别计算。输出保存完整 A、原尺寸、缩放尺寸、裁剪框、采样器与像素中心约定。第 5 步生成射线时应使用整数像素中心，与这里的内参保持一致。

外参保持输入的 w2c/c2w 约定，align_factor 原样传递；本阶段不应用尺度、不转换窗口首帧参考系、不生成 Plücker 编码。

## 输出与读取接口

成功导出首帧/文本后，会打印 `run_baseline.py --check-only` 命令，输入自动指向本次 `baseline_i2v/`，使用 BF16 I2V 配置。仅建立索引（`--export-count 0`）时，先打印保留本次窗口配置的导出命令，避免将没有图片的目录传给推理。失败时不提示下一步。命令保存于 `next_command.txt/json`，在仓库目录执行；正式 baseline 的生成尺寸/时长由其配置决定，自定义窗口尺寸/时长时应另行核对。

结果位于 `output/windows_realestate10k_<split>_f125_fps24_704x1280_seed<seed>_<时间>_<tag>/`；也可指定唯一 `--run-name`，已有目录拒绝覆盖。

| 文件 | 内容 |
| --- | --- |
| sampling.txt / resolved_config.yaml / source_config.yaml | 全部生效参数、原配置和结果摘要 |
| launch.json / status.json / summary.json / rejected.json | 软件环境、执行状态、可用数量、剔除原因和抽查范围 |
| window_index.json | 可用样本、源清单及视频哈希、时间索引路径、切镜来源 |
| timelines/*.npz | 真实时间戳、可行起点、有效区间、切镜边界及检测分数；不保存视频像素 |
| samples/<样本名>/first_frame.png、first_frame.txt | 当前窗口的条件图像和单行提示词 |
| samples/<样本名>/gt.mp4 | 按 24 FPS 编码的 125 帧窗口预览；有损编码，非精确像素缓存 |
| samples/<样本名>/camera.npz、sample.json | 对齐后的内参、原约定外参、时间戳、索引、掩码、变换参数和来源 |
| baseline_i2v/*.png、*.txt | 可直接供现有 I2V 复现入口使用的图片和同名文本 |

正常结束时 status 为 `completed` 或 `completed_with_rejections`，至少有一条可用窗口。零可用窗口或执行失败返回非零退出码；检查 rejected.json 中的原因。GT 预览为有损 H.264/YUV420 视频，first_frame.png 对应加载器中的精确处理后首帧，不用 MP4 解码像素判断逐位相等。

```python
from utils.realcam_windows import RealCamWindowDataset

dataset = RealCamWindowDataset(
    "output/<window_run>/window_index.json",
    "<workspace>/public_data/RealCam-Vid/RealEstate10K",
)
dataset.set_epoch(0)
sample = dataset[0]
# sample['video']: float32 NumPy [125,3,704,1280], 范围 [-1,1]
# sample['image']: [3,704,1280]，与 video[0] 完全一致
# sample['prompt']: 原始文本
# sample['camera_intrinsics']: [125,3,3]，像素单位
# sample['camera_extrinsics']: [125,4,4]，保留原坐标与尺度
# 还包含 align_factor、frame_indices、timestamps、target_timestamps、
# relative_timestamps、camera_timestamps、valid_mask 和 metadata。
```

若未提供原始相机时间戳，camera_timestamps 使用按帧关联的视频时间戳，并显式记录 `camera_timestamp_source=video_frame_association`。视频及时间缓存首次访问时核对哈希，之后检查文件大小/mtime；原清单或相机变化也会报错。

位于仓库内的源清单存为仓库相对路径；搬迁时保留第 3、4 步完整输出和数据相对布局。源清单原本位于仓库外时，可通过 `manifest_override` 指向搬迁后的清单。训练器接入仍在第 7 步；目前返回 NumPy，后续按需转为 Torch。使用多进程持久 worker 时，需要训练器将 epoch 传到 worker，不能只修改主进程的 dataset 属性。

完整默认窗口约占 1.26 GiB CPU 内存。建立索引不保存整段像素；导出/读取只分配一个窗口。建议先少量导出确认，再用 `--export-count 0` 建全量索引。

用导出首帧验证原 LongLive I2V：

```bash
python run_baseline.py --set data.data_path=output/<window_run>/baseline_i2v --tag window-i2v-baseline
```

多行原始 caption 会保存于 sample.json；导出的同名文本将空白折叠成单行，防止原 I2V 入口将换行当作新镜头。此基线仍只使用首帧与文本，不能据此宣称相机控制已实现。

测试：`python -m unittest discover -s tests -p test_realcam_windows.py -v`，需要 NumPy、OmegaConf、Pillow、PyAV（项目环境已有这些依赖），不需要 Torch/GPU。
