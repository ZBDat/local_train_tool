# local_train_tool

基于 RT-DETR 的目标检测训练流水线（支持 uint6 `.tif` 图像 + YOLO 单行 `.txt` 标签）。

## 功能

- 读取数据集（YOLO 目录结构）：
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`
- 将 uint6 tif 图像转换为 8-bit PNG（训练前自动完成）
- 使用 `ultralytics` 的 RT-DETR 训练
- 训练期间写入 TensorBoard（损失曲线）
- 训练期间按 `val/box_loss + val/cls_loss + val/dfl_loss` 选出并保存最优权重：
  - `best_val_loss.pt`

## 安装依赖

```bash
pip install ultralytics pillow numpy tensorboard
```

## 训练命令示例

```bash
python /home/runner/work/local_train_tool/local_train_tool/train_rtdetr.py \
  --dataset-root /path/to/dataset \
  --class-names object \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --model rtdetr-l.pt \
  --project runs/detect \
  --name rtdetr_uint6
```

## TensorBoard 监控

```bash
tensorboard --logdir runs/detect
```

然后在浏览器打开输出地址查看训练/验证损失曲线。
