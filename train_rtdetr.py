#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import urllib.request
from urllib.error import URLError
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from ultralytics import RTDETR


COCO_PRETRAINED_WEIGHTS = {
    "coco-rtdetr-l": "https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-l.pt",
    "coco-rtdetr-x": "https://github.com/ultralytics/assets/releases/download/v8.0.0/rtdetr-x.pt",
}


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


def convert_tifs_to_float32(dataset_root: Path, reuse_prepared: bool = False, force_rebuild_prepared: bool = False) -> Path:
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

    for split in ("train", "val"):
        image_dir = prepared_root / "images" / split
        if not image_dir.exists():
            continue
        for tif_path in list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")):
            with Image.open(tif_path) as img:
                arr = np.array(img)
            arr_f32 = _convert_to_float32_single_channel(arr)
            save_path = tif_path.with_suffix(".tif")
            Image.fromarray(arr_f32, mode="F").save(save_path)
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

    prepared_root = convert_tifs_to_float32(
        args.dataset_root,
        reuse_prepared=args.reuse_prepared,
        force_rebuild_prepared=args.force_rebuild_prepared,
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
