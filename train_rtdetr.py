#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on uint6 tif + YOLO labels dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="YOLO dataset root directory.")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names, e.g. --class-names person car")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="RT-DETR pretrained weight or model yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="rtdetr_uint6")
    return parser.parse_args()


def _uint6_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        max_val = int(arr.max()) if arr.size else 0
        if max_val <= 63:
            return (arr.astype(np.uint16) * 255 // 63).astype(np.uint8)
        if max_val <= 255:
            return arr.astype(np.uint8)
        return (arr.astype(np.uint32) * 255 // max_val).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
    raise TypeError(f"Unsupported tif dtype: {arr.dtype}")


def convert_uint6_tifs(dataset_root: Path) -> Path:
    prepared_root = dataset_root.parent / f"{dataset_root.name}_prepared"
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    shutil.copytree(dataset_root, prepared_root)

    for split in ("train", "val"):
        image_dir = prepared_root / "images" / split
        if not image_dir.exists():
            continue
        for tif_path in list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")):
            with Image.open(tif_path) as img:
                arr = np.array(img)
            arr8 = _uint6_to_uint8(arr)
            png_path = tif_path.with_suffix(".png")
            Image.fromarray(arr8).save(png_path)
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
    lines = [ln.strip() for ln in results_csv.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        row = {}
        for h, p in zip(headers, parts):
            try:
                row[h] = float(p)
            except ValueError:
                row[h] = p
        rows.append(row)
    return rows


def _find_best_epoch_by_val_loss(rows: Iterable[dict]) -> int:
    best_idx = -1
    best_loss = None
    for idx, row in enumerate(rows):
        box = float(row.get("val/box_loss", np.inf))
        cls = float(row.get("val/cls_loss", np.inf))
        dfl = float(row.get("val/dfl_loss", np.inf))
        val_loss = box + cls + dfl
        if np.isfinite(val_loss) and (best_loss is None or val_loss < best_loss):
            best_loss = val_loss
            best_idx = idx
    return best_idx


def export_tensorboard_and_best(run_dir: Path) -> None:
    rows = list(_load_results_csv(run_dir))
    if not rows:
        return

    writer = SummaryWriter(log_dir=str(run_dir))
    best_epoch = _find_best_epoch_by_val_loss(rows)
    best_row = rows[best_epoch] if best_epoch >= 0 else {}
    best_loss = float("inf")

    for epoch, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float):
                writer.add_scalar(key, value, epoch)
        val_loss = (
            float(row.get("val/box_loss", np.inf))
            + float(row.get("val/cls_loss", np.inf))
            + float(row.get("val/dfl_loss", np.inf))
        )
        if np.isfinite(val_loss):
            writer.add_scalar("val/loss_total", val_loss, epoch)
        if epoch == best_epoch and np.isfinite(val_loss):
            best_loss = val_loss
    writer.flush()
    writer.close()

    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, run_dir / "weights" / "best_val_loss.pt")

    (run_dir / "best_val_loss.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": best_loss, "metrics": best_row}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    prepared_root = convert_uint6_tifs(args.dataset_root)
    data_yaml = write_data_yaml(prepared_root, args.class_names)

    model = RTDETR(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )

    run_dir = Path(args.project) / args.name
    export_tensorboard_and_best(run_dir)
    print(f"Training completed. Run dir: {run_dir}")
    print(f"Best val loss checkpoint: {run_dir / 'weights' / 'best_val_loss.pt'}")


if __name__ == "__main__":
    main()
