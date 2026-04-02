#!/usr/bin/env python3
import argparse
import csv
import json
import random
import shutil
import urllib.request
from urllib.error import URLError
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.tensorboard import SummaryWriter
from ultralytics import RTDETR


COCO_PRETRAINED_WEIGHTS = {
    "coco-rtdetr-l": "https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-l.pt",
    "coco-rtdetr-x": "https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-x.pt",
}


YoloLabel = Tuple[int, float, float, float, float]


def _val_loss_total(row: dict) -> float:
    return (
        float(row.get("val/box_loss", np.inf))
        + float(row.get("val/cls_loss", np.inf))
        + float(row.get("val/dfl_loss", np.inf))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on TIFF + YOLO labels dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="YOLO dataset root directory.")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names, e.g. --class-names person car")
    parser.add_argument(
        "--model",
        type=str,
        default="rtdetr-l.pt",
        help="RT-DETR weight/model yaml, or preset key: coco-rtdetr-l / coco-rtdetr-x.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory for downloaded preset weights.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="rtdetr_train")
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="Number of offline augmented copies to generate per train image before training.",
    )
    parser.add_argument("--augment-seed", type=int, default=42, help="Random seed for offline data augmentation.")
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Explicitly request reusing existing prepared dataset directory if it already exists (default behavior).",
    )
    parser.add_argument(
        "--force-rebuild-prepared",
        action="store_true",
        help="Force rebuild prepared dataset directory even if it already exists.",
    )
    return parser.parse_args()


def resolve_model_path(model: str, weights_dir: Path) -> str:
    if model not in COCO_PRETRAINED_WEIGHTS:
        return model

    url = COCO_PRETRAINED_WEIGHTS[model]
    weights_dir.mkdir(parents=True, exist_ok=True)
    dst = weights_dir / Path(url).name
    if not dst.exists():
        try:
            urllib.request.urlretrieve(url, dst)
        except URLError as exc:
            raise RuntimeError(f"Failed to download preset model '{model}' from {url}: {exc}") from exc
    return str(dst)


def _convert_to_float32_single_channel(arr: np.ndarray) -> np.ndarray:
    # 6-bit TIFF data is expected in [0, 63], so normalize by 63 in that case.
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise TypeError(f"Unsupported array shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        max_val = int(arr.max()) if arr.size else 0
        if max_val <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        if max_val <= 63:
            return arr.astype(np.float32) / 63.0
        return arr.astype(np.float32) / float(max_val)
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        min_val = float(finite.min())
        max_val = float(finite.max())
        if max_val <= min_val:
            return np.zeros_like(arr, dtype=np.float32)
        scaled = (np.clip(arr, min_val, max_val) - min_val) / (max_val - min_val)
        return scaled.astype(np.float32)
    raise TypeError(f"Unsupported array dtype: {arr.dtype}")


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _read_yolo_labels(label_path: Path) -> List[YoloLabel]:
    if not label_path.exists():
        return []
    labels: List[YoloLabel] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        labels.append((cls, x, y, w, h))
    return labels


def _write_yolo_labels(label_path: Path, labels: Sequence[YoloLabel]) -> None:
    content = "\n".join(
        f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in labels
    )
    label_path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _rotate_labels(labels: Sequence[YoloLabel], k: int) -> List[YoloLabel]:
    k %= 4
    if k == 0:
        return list(labels)
    out: List[YoloLabel] = []
    for cls, x, y, w, h in labels:
        if k == 1:
            out.append((cls, y, 1.0 - x, h, w))
        elif k == 2:
            out.append((cls, 1.0 - x, 1.0 - y, w, h))
        else:
            out.append((cls, 1.0 - y, x, h, w))
    return out


def _flip_labels(labels: Sequence[YoloLabel], horizontal: bool) -> List[YoloLabel]:
    if horizontal:
        return [(cls, 1.0 - x, y, w, h) for cls, x, y, w, h in labels]
    return [(cls, x, 1.0 - y, w, h) for cls, x, y, w, h in labels]


def _xyxy_from_yolo(box: YoloLabel, width: int, height: int) -> Tuple[int, float, float, float, float]:
    cls, x, y, w, h = box
    x1 = (x - w / 2.0) * width
    y1 = (y - h / 2.0) * height
    x2 = (x + w / 2.0) * width
    y2 = (y + h / 2.0) * height
    return cls, x1, y1, x2, y2


def _yolo_from_xyxy(cls: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> YoloLabel:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cls, cx / width, cy / height, bw / width, bh / height


def _random_crop_and_resize(
    img: np.ndarray, labels: Sequence[YoloLabel], rng: random.Random
) -> Tuple[np.ndarray, List[YoloLabel]]:
    h, w = img.shape
    crop_ratio = rng.uniform(0.8, 1.0)
    crop_w = max(2, int(round(w * crop_ratio)))
    crop_h = max(2, int(round(h * crop_ratio)))
    max_x0 = max(0, w - crop_w)
    max_y0 = max(0, h - crop_h)
    x0 = rng.randint(0, max_x0) if max_x0 > 0 else 0
    y0 = rng.randint(0, max_y0) if max_y0 > 0 else 0
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    crop = img[y0:y1, x0:x1]
    resized = np.array(
        Image.fromarray(crop, mode="F").resize((w, h), resample=Image.BILINEAR),
        dtype=np.float32,
    )

    scale_x = w / crop_w
    scale_y = h / crop_h
    new_labels: List[YoloLabel] = []
    for box in labels:
        cls, bx1, by1, bx2, by2 = _xyxy_from_yolo(box, w, h)
        ix1 = max(bx1, x0)
        iy1 = max(by1, y0)
        ix2 = min(bx2, x1)
        iy2 = min(by2, y1)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        rx1 = (ix1 - x0) * scale_x
        ry1 = (iy1 - y0) * scale_y
        rx2 = (ix2 - x0) * scale_x
        ry2 = (iy2 - y0) * scale_y
        new_box = _yolo_from_xyxy(cls, rx1, ry1, rx2, ry2, w, h)
        if new_box[3] <= 0 or new_box[4] <= 0:
            continue
        new_labels.append(new_box)
    return resized, new_labels


def _random_rotate_90(img: np.ndarray, labels: Sequence[YoloLabel], rng: random.Random) -> Tuple[np.ndarray, List[YoloLabel]]:
    k = rng.choice([1, 2, 3])
    return np.rot90(img, k=k).astype(np.float32), _rotate_labels(labels, k)


def _random_flip(img: np.ndarray, labels: Sequence[YoloLabel], rng: random.Random) -> Tuple[np.ndarray, List[YoloLabel]]:
    horizontal = rng.random() < 0.5
    flipped = np.fliplr(img) if horizontal else np.flipud(img)
    return flipped.astype(np.float32), _flip_labels(labels, horizontal)


def _random_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    factor = rng.uniform(0.7, 1.3)
    mean = float(img.mean())
    return _clip01((img - mean) * factor + mean)


def _random_brightness(img: np.ndarray, rng: random.Random) -> np.ndarray:
    factor = rng.uniform(0.7, 1.3)
    return _clip01(img * factor)


def _random_gamma(img: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma = rng.uniform(0.7, 1.5)
    return _clip01(np.power(img, gamma))


def _random_gaussian_noise(img: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(0.0, 0.03)
    noise = np_rng.normal(0.0, sigma, size=img.shape)
    return _clip01(img + noise)


def _random_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    radius = rng.uniform(0.0, 1.2)
    blurred = Image.fromarray(img, mode="F").filter(ImageFilter.GaussianBlur(radius=radius))
    return _clip01(np.array(blurred, dtype=np.float32))


def _apply_random_augmentations(
    img_f32: np.ndarray, labels: Sequence[YoloLabel], rng: random.Random, np_rng: np.random.Generator
) -> Tuple[np.ndarray, List[YoloLabel]]:
    img = img_f32.copy()
    aug_labels = list(labels)

    if rng.random() < 0.6:
        img, aug_labels = _random_crop_and_resize(img, aug_labels, rng)
    if rng.random() < 0.6:
        img, aug_labels = _random_rotate_90(img, aug_labels, rng)
    if rng.random() < 0.5:
        img, aug_labels = _random_flip(img, aug_labels, rng)

    if rng.random() < 0.7:
        img = _random_contrast(img, rng)
    if rng.random() < 0.7:
        img = _random_brightness(img, rng)
    if rng.random() < 0.5:
        img = _random_gamma(img, rng)
    if rng.random() < 0.4:
        img = _random_gaussian_noise(img, rng, np_rng)
    if rng.random() < 0.4:
        img = _random_blur(img, rng)

    return _clip01(img), aug_labels


def _check_uint16_augmentation_compatibility(seed: int = 42) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sample = np.linspace(0, 65535, 128 * 128).reshape(128, 128).astype(np.uint16)
    base_img = _convert_to_float32_single_channel(sample)
    labels: List[YoloLabel] = [(0, 0.5, 0.5, 0.4, 0.4)]

    checks = [
        ("random_crop", lambda: _random_crop_and_resize(base_img, labels, rng)),
        ("random_rotation", lambda: _random_rotate_90(base_img, labels, rng)),
        ("flip", lambda: _random_flip(base_img, labels, rng)),
        ("random_contrast", lambda: (_random_contrast(base_img, rng), labels)),
        ("random_brightness", lambda: (_random_brightness(base_img, rng), labels)),
        ("random_gamma", lambda: (_random_gamma(base_img, rng), labels)),
        ("gaussian_noise", lambda: (_random_gaussian_noise(base_img, rng, np_rng), labels)),
        ("gaussian_blur", lambda: (_random_blur(base_img, rng), labels)),
        ("combined_pipeline", lambda: _apply_random_augmentations(base_img, labels, rng, np_rng)),
    ]
    for aug_name, fn in checks:
        out_img, out_labels = fn()
        if out_img.ndim != 2 or not np.isfinite(out_img).all():
            raise RuntimeError(
                f"uint16 TIFF augmentation compatibility check failed in '{aug_name}': invalid image output."
            )
        for _, x, y, w, h in out_labels:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0.0 and h > 0.0 and w <= 1.0 and h <= 1.0):
                raise RuntimeError(
                    f"uint16 TIFF augmentation compatibility check failed in '{aug_name}': "
                    f"invalid bbox (x={x}, y={y}, w={w}, h={h})."
                )


def convert_tifs_to_float32(
    dataset_root: Path,
    reuse_prepared: bool = False,
    force_rebuild_prepared: bool = False,
    augment_copies: int = 0,
    augment_seed: int = 42,
) -> Path:
    prepared_root = dataset_root.parent / f"{dataset_root.name}_prepared"
    if reuse_prepared and force_rebuild_prepared:
        raise ValueError("--reuse-prepared and --force-rebuild-prepared cannot be used together.")
    if prepared_root.exists():
        if reuse_prepared:
            return prepared_root
        if force_rebuild_prepared:
            try:
                shutil.rmtree(prepared_root)
            except OSError as exc:
                raise RuntimeError(f"Failed to clean prepared dataset directory: {prepared_root}") from exc
        else:
            return prepared_root
    shutil.copytree(dataset_root, prepared_root)

    rng = random.Random(augment_seed)
    np_rng = np.random.default_rng(augment_seed)
    for split in ("train", "val"):
        image_dir = prepared_root / "images" / split
        label_dir = prepared_root / "labels" / split
        if not image_dir.exists():
            continue
        for tif_path in list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")):
            with Image.open(tif_path) as img:
                arr = np.array(img)
            arr_f32 = _convert_to_float32_single_channel(arr)
            save_path = tif_path.with_suffix(".tif")
            Image.fromarray(arr_f32, mode="F").save(save_path)
            if split == "train" and augment_copies > 0:
                base_labels = _read_yolo_labels(label_dir / f"{tif_path.stem}.txt")
                for idx in range(augment_copies):
                    aug_img, aug_labels = _apply_random_augmentations(arr_f32, base_labels, rng, np_rng)
                    aug_stem = f"{save_path.stem}_aug{idx + 1}"
                    Image.fromarray(aug_img, mode="F").save(image_dir / f"{aug_stem}.tif")
                    _write_yolo_labels(label_dir / f"{aug_stem}.txt", aug_labels)
            if tif_path != save_path:
                tif_path.unlink()
    return prepared_root


def write_data_yaml(dataset_root: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    data_yaml = dataset_root / "rtdetr_data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: {}\nval: {}\nnames:\n{}\n".format(
            data["path"],
            data["train"],
            data["val"],
            "\n".join([f"  {i}: {name}" for i, name in data["names"].items()]),
        ),
        encoding="utf-8",
    )
    return data_yaml


def _load_results_csv(run_dir: Path) -> Iterable[dict]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return []
    rows = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {}
            for h, p in raw_row.items():
                # csv.DictReader sets key=None when a row has more fields than headers.
                if h is None:
                    continue
                try:
                    row[h] = float(p)
                except (TypeError, ValueError):
                    row[h] = p
            if row:
                rows.append(row)
    return rows


def _find_best_epoch_by_val_loss(rows: Iterable[dict]) -> int:
    best_idx = -1
    best_loss = None
    for idx, row in enumerate(rows):
        val_loss = _val_loss_total(row)
        if np.isfinite(val_loss) and (best_loss is None or val_loss < best_loss):
            best_loss = val_loss
            best_idx = idx
    return best_idx


def export_tensorboard_and_best(run_dir: Path) -> None:
    rows = list(_load_results_csv(run_dir))
    if not rows:
        return

    best_epoch = _find_best_epoch_by_val_loss(rows)
    best_row = rows[best_epoch] if best_epoch >= 0 else {}
    best_loss = float("inf")

    for epoch, row in enumerate(rows):
        val_loss = _val_loss_total(row)
        if epoch == best_epoch and np.isfinite(val_loss):
            best_loss = val_loss

    best_src = run_dir / "weights" / f"epoch{best_epoch}.pt"
    if not best_src.exists():
        # Ultralytics epoch checkpoint naming can be 0-based or 1-based across versions.
        alt_src = run_dir / "weights" / f"epoch{best_epoch + 1}.pt"
        if alt_src.exists():
            best_src = alt_src
    if not best_src.exists():
        best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, run_dir / "weights" / "best_val_loss.pt")

    (run_dir / "best_val_loss.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": best_loss, "metrics": best_row}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_run_dir(train_result, model: RTDETR, default_run_dir: Path) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None) if trainer is not None else None
    return Path(save_dir) if save_dir else default_run_dir


def main() -> None:
    args = parse_args()
    if args.augment_copies < 0:
        raise ValueError("--augment-copies must be >= 0")

    _check_uint16_augmentation_compatibility(seed=args.augment_seed)
    print("uint16 TIFF augmentation compatibility check passed.")

    prepared_root = convert_tifs_to_float32(
        args.dataset_root,
        reuse_prepared=args.reuse_prepared,
        force_rebuild_prepared=args.force_rebuild_prepared,
        augment_copies=args.augment_copies,
        augment_seed=args.augment_seed,
    )
    data_yaml = write_data_yaml(prepared_root, args.class_names)
    model_path = resolve_model_path(args.model, args.weights_dir)

    model = RTDETR(model_path)
    default_run_dir = Path(args.project) / args.name
    tb_writer = None

    def _get_tb_writer(trainer) -> SummaryWriter:
        nonlocal tb_writer
        if tb_writer is None:
            trainer_run_dir = getattr(trainer, "save_dir", None)
            log_dir = Path(trainer_run_dir) if trainer_run_dir else default_run_dir
            tb_writer = SummaryWriter(log_dir=str(log_dir))
        return tb_writer

    def _close_tb_writer() -> None:
        nonlocal tb_writer
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
            tb_writer = None

    def on_fit_epoch_end(trainer) -> None:
        writer = _get_tb_writer(trainer)
        metrics = getattr(trainer, "metrics", {})
        epoch = int(getattr(trainer, "epoch", 0))
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(str(key), float(value), epoch)
        val_loss = _val_loss_total(metrics)
        if np.isfinite(val_loss):
            writer.add_scalar("val/loss_total", val_loss, epoch)

    def on_train_end(trainer) -> None:
        _close_tb_writer()

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    try:
        train_result = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            save_period=1,
        )
    finally:
        _close_tb_writer()

    run_dir = _resolve_run_dir(train_result, model, default_run_dir)
    export_tensorboard_and_best(run_dir)
    print(f"Training completed. Run dir: {run_dir}")
    print(f"Best val loss checkpoint: {run_dir / 'weights' / 'best_val_loss.pt'}")


if __name__ == "__main__":
    main()
