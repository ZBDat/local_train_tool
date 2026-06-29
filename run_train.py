#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_rtdetr.py with recommended default arguments."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("./dataset"), help="Dataset root path.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["object"],
        help="Class names for training, e.g. --class-names person car",
    )
    parser.add_argument("--model", type=str, default="coco-rtdetr-l", help="Model preset or weight path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Training project directory.")
    parser.add_argument("--name", type=str, default="rtdetr_recommended", help="Run name.")
    parser.add_argument("--augment-copies", type=int, default=1, help="Offline augmentation copies per train image.")
    parser.add_argument("--augment-seed", type=int, default=42, help="Offline augmentation random seed.")
    parser.add_argument("--augment-mosaic-prob", type=float, default=0.35)
    parser.add_argument("--augment-translate-scale-prob", type=float, default=0.6)
    parser.add_argument("--augment-cutout-prob", type=float, default=0.45)
    parser.add_argument("--augment-clahe-prob", type=float, default=0.45)
    parser.add_argument("--augment-gamma-prob", type=float, default=0.5)
    parser.add_argument("--augment-hist-perturb-prob", type=float, default=0.5)
    parser.add_argument("--augment-blur-noise-combo-prob", type=float, default=0.35)
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=("per_image", "fixed_uint16"),
        default="per_image",
        help="Image normalization mode.",
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"), help="Preset weight cache directory.")
    parser.add_argument("--preset-sha256", type=str, default="", help="Optional expected preset SHA256.")
    parser.add_argument("--reuse-prepared", action="store_true", help="Reuse prepared dataset if it exists.")
    parser.add_argument("--force-rebuild-prepared", action="store_true", help="Force rebuild prepared dataset.")
    parser.add_argument("--raw-split-ratio", type=float, default=0.8, help="Train split ratio for raw flat layout.")
    parser.add_argument("--raw-split-seed", type=int, default=42, help="Random seed for raw split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "train_rtdetr.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset-root",
        str(args.dataset_root),
        "--class-names",
        *args.class_names,
        "--model",
        args.model,
        "--weights-dir",
        str(args.weights_dir),
        "--preset-sha256",
        args.preset_sha256,
        "--normalize-mode",
        args.normalize_mode,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--project",
        args.project,
        "--name",
        args.name,
        "--augment-copies",
        str(args.augment_copies),
        "--augment-seed",
        str(args.augment_seed),
        "--augment-mosaic-prob",
        str(args.augment_mosaic_prob),
        "--augment-translate-scale-prob",
        str(args.augment_translate_scale_prob),
        "--augment-cutout-prob",
        str(args.augment_cutout_prob),
        "--augment-clahe-prob",
        str(args.augment_clahe_prob),
        "--augment-gamma-prob",
        str(args.augment_gamma_prob),
        "--augment-hist-perturb-prob",
        str(args.augment_hist_perturb_prob),
        "--augment-blur-noise-combo-prob",
        str(args.augment_blur_noise_combo_prob),
        "--raw-split-ratio",
        str(args.raw_split_ratio),
        "--raw-split-seed",
        str(args.raw_split_seed),
    ]
    if args.reuse_prepared:
        cmd.append("--reuse-prepared")
    if args.force_rebuild_prepared:
        cmd.append("--force-rebuild-prepared")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
