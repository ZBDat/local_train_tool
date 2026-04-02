# local_train_tool

基于 RT-DETR 的目标检测训练流水线（支持 6-bit 数值范围的 `.tif` 图像 + YOLO 单行 `.txt` 标签）。

## 功能

- 读取数据集（YOLO 目录结构）：
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`
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
- 使用 `ultralytics` 的 RT-DETR 训练
- 支持公开 COCO 预训练权重快捷选项（会自动下载）：
  - `coco-rtdetr-l`（RT-DETR-L）
  - `coco-rtdetr-x`（RT-DETR-X）
- 训练期间写入 TensorBoard（损失曲线）
- 训练期间按 `val/box_loss + val/cls_loss + val/dfl_loss` 选出并保存最优权重：
  - `best_val_loss.pt`

## 安装依赖

```bash
pip install ultralytics pillow numpy tensorboard
```

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
- 也可以继续用本地模型路径或 `rtdetr-l.pt` / `rtdetr-x.pt`：
  - `--model /path/to/your_model.pt`
  - `--model rtdetr-l.pt`
- 当 `--model` 取 `coco-rtdetr-l` 或 `coco-rtdetr-x` 时，脚本会从公开地址下载权重到 `weights/`（可用 `--weights-dir` 修改）。
- 数据预处理目录默认为 `<dataset-root>_prepared`：
  - 若该目录已存在，默认直接复用；
  - 可使用 `--force-rebuild-prepared` 强制重建；
  - 可使用 `--reuse-prepared` 显式声明复用（与 `--force-rebuild-prepared` 互斥）。
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

## TensorBoard 监控

```bash
tensorboard --logdir runs/detect
```

然后在浏览器打开输出地址查看训练/验证损失曲线。
