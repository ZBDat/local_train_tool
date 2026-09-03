# local_train_tool

通用目标检测训练流水线（支持 6-bit 数值范围的 `.tif` 图像 + YOLO 单行 `.txt` 标签）。

提供两套训练入口：

- `train_rtdetr.py`：基于 Ultralytics 训练接口（RT-DETR / YOLO），预处理阶段会把数据量化后交给 Ultralytics 管线。
- `train_monai_rtdetr.py`：**MONAI 16 位数据管线 + RT-DETR**，全程保留 uint16 位深（float32 [0,1] 输入模型），绕开 Ultralytics 训练管线中硬编码的 8-bit 假设（mosaic 画布 uint8、`/255` 归一化等）。适合需要真正保留高位深数据的训练场景。

## 功能

- 读取数据集（支持两种输入结构）：
  - YOLO 结构：`images/train`, `images/val`, `labels/train`, `labels/val`
  - 原始扁平结构：`images/*.tif(f)` + `yolo_annotations/*.txt`（脚本会自动切分并转换为 YOLO 结构）
- 支持无标注样本：若某张图缺少对应 `.txt`，会自动创建空标签文件（表示该图无目标）
- 将 TIFF 图像转换为 **float32 单通道 TIFF**（不使用 8-bit PNG；当像素范围为 0~63 时按 6-bit 线性归一化）
- 训练前对训练集支持离线数据增强（可配置增强副本数）：
  - Mosaic（多图拼接小目标增强）
  - random crop（裁剪后缩放回原尺寸）
  - random rotation（90/180/270）
  - flip（水平/垂直）
  - 轻度平移 + 缩放仿射
  - 随机局部遮挡（Cutout）
  - 随机对比度变化
  - 随机亮度变化
  - CLAHE-like 局部对比度增强
  - 随机 gamma 变化
  - 随机直方图扰动（分位拉伸+强度扰动）
  - 随机高斯噪声
  - 随机高斯模糊
  - 轻度模糊+噪声联合退化
- 自动执行 uint16 TIFF 增强兼容性检查（启动训练前）
- 使用 `ultralytics` 的通用检测训练接口（RT-DETR / YOLO）
- 支持公开 COCO 预训练权重快捷选项：
  - `coco-rtdetr-l`（RT-DETR-L）
  - `coco-rtdetr-x`（RT-DETR-X）
  - `coco-yolo11-l`（YOLO11-L）
  - `coco-yolo11-x`（YOLO11-X）
  - `coco-yolov8-x`（YOLOv8-X）
  - `coco-deformable-detr-l`（Deformable DETR-L）
  - `coco-deformable-detr-x`（Deformable DETR-X）
  - `coco-dino-l`（DINO-L）
  - `coco-dino-x`（DINO-X）
  - `coco-nino-l`（NINO-L，当前映射为 DINO-L）
  - `coco-nino-x`（NINO-X，当前映射为 DINO-X）
- 训练期间写入 TensorBoard（损失曲线）
- 训练期间按 `val/box_loss + val/cls_loss + val/dfl_loss` 选出并保存最优权重：
  - `best_val_loss.pt`

## 安装依赖

```bash
pip install ultralytics pillow numpy tensorboard monai itk
```

`train_monai_rtdetr.py` 依赖 `monai` 与 `itk`（MONAI 用 ITK 读取 16 位 TIFF）。若 MONAI 加载失败会自动回退到 Pillow 读取（同样保留 uint16）。

## 训练命令示例

```bash
python train_rtdetr.py \
  --dataset-root /path/to/dataset \
  --class-names object \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --augment-copies 1 \
  --augment-seed 42 \
  --augment-mosaic-prob 0.35 \
  --augment-translate-scale-prob 0.6 \
  --augment-cutout-prob 0.45 \
  --augment-clahe-prob 0.45 \
  --augment-gamma-prob 0.5 \
  --augment-hist-perturb-prob 0.5 \
  --augment-blur-noise-combo-prob 0.35 \
  --model coco-rtdetr-l \
  --project runs/detect \
  --name rtdetr_train
```

说明：
- 可使用本地模型路径或官方权重名（例如：`rtdetr-l.pt`、`rtdetr-x.pt`、`yolo11x.pt`、`yolov8x.pt`）：
  - `--model /path/to/your_model.pt`
  - `--model rtdetr-l.pt`
- 当 `--model` 取 `coco-rtdetr-l` 或 `coco-rtdetr-x` 时，脚本会从公开地址下载权重到 `weights/`（可用 `--weights-dir` 修改）；下载仅允许 GitHub 相关可信域名（含重定向目标），并会校验目标文件名及落盘路径安全性。
- 对于 URL 下载的 preset 权重，可通过 `--preset-sha256 <sha256>` 启用文件完整性校验（建议在生产训练时提供）。
- 当 `--model` 取 `coco-yolo11-l` / `coco-yolo11-x` / `coco-yolov8-x` / `coco-deformable-detr-l` / `coco-deformable-detr-x` / `coco-dino-l` / `coco-dino-x` / `coco-nino-l` / `coco-nino-x` 时，直接使用 Ultralytics 内置模型名加载对应 COCO 预训练权重。
- 数据预处理目录默认为 `<dataset-root>_prepared`：
  - 若该目录已存在，默认直接复用；
  - 可使用 `--force-rebuild-prepared` 强制重建；
  - 可使用 `--reuse-prepared` 显式声明复用（与 `--force-rebuild-prepared` 互斥）。
- 若输入为原始扁平结构，会先自动生成 `<dataset-root>_yolo`：
  - `--raw-split-ratio`：训练集占比（默认 `0.8`）
  - `--raw-split-seed`：切分随机种子（默认 `42`）
- 增强参数：
  - `--augment-copies`：每张训练图像生成多少份离线增强样本（默认 0，即不额外生成）。
  - `--augment-seed`：离线增强随机种子（默认 42）。
  - 方案2（Mosaic）：`--augment-mosaic-prob`（默认 `0.35`）
  - 方案3（平移+缩放仿射）：`--augment-translate-scale-prob`（默认 `0.6`）
  - 方案4（Cutout）：`--augment-cutout-prob`（默认 `0.45`）
  - 方案5（CLAHE-like）：`--augment-clahe-prob`（默认 `0.45`）
  - 方案6（模糊+噪声联合）：`--augment-blur-noise-combo-prob`（默认 `0.35`）
  - 方案7（Gamma 与直方图扰动）：
    - `--augment-gamma-prob`（默认 `0.5`）
    - `--augment-hist-perturb-prob`（默认 `0.5`）
- 归一化策略（`--normalize-mode`）：
  - `per_image`（默认）：按像素位深上限归一化（优先保持物理强度比例；`<=63` 按 6-bit，`uint16` 按 65535，`uint8` 按 255）。
  - `fixed_6bit`：固定按 `63` 归一化。
  - `fixed_uint16`：固定按 `65535` 归一化。

## TensorBoard 监控

```bash
tensorboard --logdir runs/detect
```

然后在浏览器打开输出地址查看训练/验证损失曲线。

## MONAI 16 位训练（train_monai_rtdetr.py）

```bash
python train_monai_rtdetr.py \
  --dataset-root /path/to/dataset \
  --class-names object \
  --model weights/rtdetr-l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --normalize-mode per_image \
  --device 0 \
  --project runs/detect \
  --name monai_rtdetr
```

数据流：`uint16 TIFF → MONAI LoadImage(ITK) → ScaleIntensityRange/[0,1] float32 → 实时 box 同步增强 → letterbox（pad=0.447）→ RT-DETR 自定义训练循环`。

说明：
- `--model` 支持本地 RT-DETR 权重或 preset（`coco-rtdetr-l` / `coco-rtdetr-x`）。
- 直接读取原始 `uint16` TIFF，**不做** 8-bit 量化；输出由模型第一个卷积自动 patch 成 1 通道（RGB 均值初始化后全量微调）。
- 增强在管线内实时进行（复用 `train_rtdetr.py` 的 box 同步增强函数），不产生额外文件。
- 训练开始前自动保存 `train_batch_preview.jpg` / `val_batch_preview.jpg`（与 Ultralytics `train_batch*.jpg` 同款网格图，含 GT 框标注），方便核对数据加载与增强效果；可用 `--no-plot-batch` 关闭。
- `--pad-val`：letterbox 灰边值，默认 `0.447`（=114/255，对齐 COCO 预训练填充语义）。
- `--val-metric map`：按 mAP50-95 选优存 `best.pt`；`--val-metric loss` 则按 val 总损失选优。
- 训练产物：`best.pt` / `last.pt` / `best_val_loss.json` / `results.csv` / TensorBoard 事件。
- `--resume <last.pt>`：从断点继续训练。
- 空目标 batch 自动跳过（RT-DETR denoising 需要 GT）；单卡训练（多卡 DDP 暂未支持）。
