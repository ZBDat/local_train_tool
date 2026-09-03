#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import numbers
import random
import re
import shutil
import urllib.request
import warnings
from collections import OrderedDict
from urllib.error import URLError
from urllib.parse import urlparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import RTDETR, YOLO


MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "coco-rtdetr-l": {
        "family": "rtdetr",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt",
    },
    "coco-rtdetr-x": {
        "family": "rtdetr",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-x.pt",
    },
    "coco-yolo11-l": {
        "family": "yolo",
        "model": "yolo11l.pt",
    },
    "coco-yolo11-x": {
        "family": "yolo",
        "model": "yolo11x.pt",
    },
    "coco-yolov8-x": {
        "family": "yolo",
        "model": "yolov8x.pt",
    },
    "coco-deformable-detr-l": {
        "family": "yolo",
        "model": "deformable-detr-l.pt",
    },
    "coco-deformable-detr-x": {
        "family": "yolo",
        "model": "deformable-detr-x.pt",
    },
    "coco-dino-l": {
        "family": "yolo",
        "model": "dino-l.pt",
    },
    "coco-dino-x": {
        "family": "yolo",
        "model": "dino-x.pt",
    },
    # NINO preset aliases currently map to DINO checkpoints.
    "coco-nino-l": {
        "family": "yolo",
        "model": "dino-l.pt",
    },
    "coco-nino-x": {
        "family": "yolo",
        "model": "dino-x.pt",
    },
}
MODEL_PRESET_KEYS_TEXT = " / ".join(sorted(MODEL_PRESETS.keys()))
TRUSTED_WEIGHT_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
    "release-assets.githubusercontent.com",
}
SAFE_WEIGHT_FILENAME = re.compile(r"^[A-Za-z0-9._-]+\.pt$")
SAFE_SHA256_HEX = re.compile(r"^[A-Fa-f0-9]{64}$")
NORMALIZE_MODES = ("per_image", "fixed_uint16")
MAX_AUG_IMAGE_CACHE = 16


# YOLO format: (class_id, center_x, center_y, width, height), where x/y/w/h are normalized to [0, 1].
YoloLabel = Tuple[int, float, float, float, float]
MIN_CUTOUT_VISIBLE_RATIO = 0.35
UINT16_CHECK_IMAGE_SIZE = 128


def _val_loss_total(row: dict) -> float:
    return (
        float(row.get("val/box_loss", np.inf))
        + float(row.get("val/cls_loss", np.inf))
        + float(row.get("val/dfl_loss", np.inf))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train object detection models on TIFF + YOLO labels dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="YOLO dataset root directory.")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names, e.g. --class-names person car")
    parser.add_argument(
        "--model",
        type=str,
        default="rtdetr-l.pt",
        help=f"Model weight/model yaml path, or preset key: {MODEL_PRESET_KEYS_TEXT}.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory for downloaded preset weights.",
    )
    parser.add_argument(
        "--preset-sha256",
        type=str,
        default="",
        help="Optional expected SHA256 for downloaded URL-based preset weights.",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=NORMALIZE_MODES,
        default="per_image",
        help="Image normalization mode: per_image / fixed_uint16.",
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
    parser.add_argument("--augment-mosaic-prob", type=float, default=0.35, help="Probability of mosaic augmentation.")
    parser.add_argument(
        "--augment-translate-scale-prob",
        type=float,
        default=0.6,
        help="Probability of translate+scale affine augmentation.",
    )
    parser.add_argument("--augment-cutout-prob", type=float, default=0.45, help="Probability of cutout augmentation.")
    parser.add_argument(
        "--augment-clahe-prob",
        type=float,
        default=0.45,
        help="Probability of CLAHE-like local contrast augmentation.",
    )
    parser.add_argument("--augment-gamma-prob", type=float, default=0.5, help="Probability of gamma augmentation.")
    parser.add_argument(
        "--augment-hist-perturb-prob",
        type=float,
        default=0.5,
        help="Probability of histogram perturbation augmentation.",
    )
    parser.add_argument(
        "--augment-blur-noise-combo-prob",
        type=float,
        default=0.35,
        help="Probability of blur+noise combo augmentation.",
    )
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
    parser.add_argument(
        "--raw-split-ratio",
        type=float,
        default=0.8,
        help="Train split ratio when auto-converting flat raw layout (images + yolo_annotations).",
    )
    parser.add_argument(
        "--raw-split-seed",
        type=int,
        default=42,
        help="Random seed used for auto split when converting flat raw layout.",
    )
    return parser.parse_args()


def _infer_model_family(model_ref: str) -> str:
    lowered_model_ref = model_ref.lower()
    if "rtdetr" in lowered_model_ref:
        return "rtdetr"
    if any(token in lowered_model_ref for token in ("deformable", "dino", "nino")):
        return "yolo"
    return "yolo"


def _sha256_of_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_expected_sha256(file_path: Path, expected_sha256: str) -> None:
    expected = expected_sha256.strip().lower()
    if not expected:
        return
    if not SAFE_SHA256_HEX.fullmatch(expected):
        raise ValueError(f"Invalid SHA256 format: '{expected_sha256}'")
    actual = _sha256_of_file(file_path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for '{file_path.name}': expected {expected}, got {actual}."
        )


def _safe_download_preset_weight(url: str, weights_dir: Path, expected_sha256: str = "") -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only https URLs are allowed for preset weights: {url}")
    if not parsed.hostname or parsed.hostname.lower() not in TRUSTED_WEIGHT_HOSTS:
        raise ValueError(f"Untrusted weight host in preset URL: {url}")
    filename = Path(parsed.path).name
    if not SAFE_WEIGHT_FILENAME.fullmatch(filename):
        raise ValueError(f"Unsafe preset weight filename: {filename}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    root = weights_dir.resolve()
    dst = (weights_dir / filename).resolve()
    if root != dst.parent:
        raise RuntimeError(f"Invalid preset weight destination path: {dst}")

    if dst.exists():
        _verify_expected_sha256(dst, expected_sha256)
        return dst

    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            final_url = response.geturl()
            final_parsed = urlparse(final_url)
            if not final_parsed.hostname or final_parsed.hostname.lower() not in TRUSTED_WEIGHT_HOSTS:
                raise RuntimeError(f"Preset weight redirected to untrusted host: {final_url}")

            expected = expected_sha256.strip().lower()
            if expected and not SAFE_SHA256_HEX.fullmatch(expected):
                raise ValueError(f"Invalid SHA256 format: '{expected_sha256}'")
            hasher = hashlib.sha256()
            with tmp_dst.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)

            if expected and hasher.hexdigest() != expected:
                try:
                    tmp_dst.unlink(missing_ok=True)
                except OSError:
                    pass
                raise RuntimeError(
                    f"SHA256 mismatch for downloaded '{filename}': expected {expected}, got {hasher.hexdigest()}."
                )
            tmp_dst.replace(dst)
    except URLError as exc:
        raise RuntimeError(f"Failed to download preset model from {url}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to save preset model to '{dst}': {exc}") from exc

    _verify_expected_sha256(dst, expected_sha256)
    return dst


def resolve_model_path(model: str, weights_dir: Path, preset_sha256: str = "") -> Tuple[str, str]:
    if model not in MODEL_PRESETS:
        return model, _infer_model_family(model)

    preset = MODEL_PRESETS[model]
    family = preset["family"]
    if "url" in preset:
        expected_sha256 = preset_sha256.strip() or preset.get("sha256", "")
        dst = _safe_download_preset_weight(preset["url"], weights_dir, expected_sha256=expected_sha256)
        return str(dst), family
    return preset["model"], family


def _convert_to_float32_single_channel(arr: np.ndarray, normalize_mode: str = "per_image") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise TypeError(f"Unsupported array shape: {arr.shape}")
    if normalize_mode not in NORMALIZE_MODES:
        raise ValueError(f"Unsupported normalize mode: {normalize_mode}")

    if normalize_mode == "fixed_uint16":
        return _clip01(arr.astype(np.float32) / 65535.0)

    if np.issubdtype(arr.dtype, np.integer):
        max_val = int(arr.max()) if arr.size else 0
        if max_val <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        arr_nonneg = np.clip(arr, 0, None).astype(np.float64, copy=False)
        dtype_max = int(np.iinfo(arr.dtype).max)
        if arr.dtype == np.uint8:
            return _clip01((arr_nonneg / 255.0).astype(np.float32))
        if dtype_max > 0:
            return _clip01((arr_nonneg / float(dtype_max)).astype(np.float32))
        return _clip01((arr_nonneg / float(max_val)).astype(np.float32))

    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        min_val = float(finite.min())
        max_val = float(finite.max())
        if max_val <= 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        clipped = np.clip(arr, 0.0, None)
        if max_val <= 1.0:
            return _clip01(clipped)
        if max_val <= 255.0:
            return _clip01((clipped / 255.0).astype(np.float32))
        if max_val <= 65535.0:
            return _clip01((clipped / 65535.0).astype(np.float32))
        return _clip01((clipped / max_val).astype(np.float32))

    raise TypeError(f"Unsupported array dtype: {arr.dtype}")


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _read_tiff_array_lossless(tif_path: Path) -> np.ndarray:
    """Read TIFF from bytes and preserve source bit-depth as much as possible."""
    raw = tif_path.read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        mode = img.mode
        arr = np.asarray(img)

    if arr.dtype == np.uint8 and mode.startswith("I;16"):
        raise RuntimeError(
            f"Failed to decode 16-bit TIFF without downcast at '{tif_path}'. "
            "Please check Pillow/TIFF codec support in current environment."
        )
    return np.array(arr, copy=True)


def _sanitize_yolo_box(cls: int, x: float, y: float, w: float, h: float) -> YoloLabel | None:
    vals = (x, y, w, h)
    if not all(np.isfinite(v) for v in vals):
        return None
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    if w <= 0.0 or h <= 0.0:
        return None
    return cls, x, y, w, h


def _read_yolo_labels(label_path: Path) -> List[YoloLabel]:
    if not label_path.exists():
        return []
    labels: List[YoloLabel] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            warnings.warn(
                f"Skipping malformed YOLO label line at {label_path}:{line_no} (expected 5 fields).",
                RuntimeWarning,
            )
            continue
        try:
            cls = int(parts[0])
        except (TypeError, ValueError):
            try:
                cls = int(float(parts[0]))
            except (TypeError, ValueError):
                warnings.warn(
                    f"Skipping YOLO label line with invalid class id at {label_path}:{line_no}.",
                    RuntimeWarning,
                )
                continue

        try:
            x, y, w, h = map(float, parts[1:])
        except (TypeError, ValueError):
            warnings.warn(
                f"Skipping YOLO label line with invalid bbox values at {label_path}:{line_no}.",
                RuntimeWarning,
            )
            continue

        sanitized = _sanitize_yolo_box(cls, x, y, w, h)
        if sanitized is None:
            warnings.warn(
                f"Skipping YOLO label line with non-positive or non-finite bbox at {label_path}:{line_no}.",
                RuntimeWarning,
            )
            continue
        labels.append(sanitized)
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


def _random_translate_scale(
    img: np.ndarray, labels: Sequence[YoloLabel], rng: random.Random
) -> Tuple[np.ndarray, List[YoloLabel]]:
    h, w = img.shape
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-0.08 * w, 0.08 * w)
    ty = rng.uniform(-0.08 * h, 0.08 * h)

    a = 1.0 / scale
    e = 1.0 / scale
    c = -tx / scale
    f = -ty / scale
    warped = Image.fromarray(img, mode="F").transform(
        (w, h),
        Image.AFFINE,
        data=(a, 0.0, c, 0.0, e, f),
        resample=Image.BILINEAR,
        fillcolor=0.0,
    )
    out = np.array(warped, dtype=np.float32)

    new_labels: List[YoloLabel] = []
    for box in labels:
        cls, x1, y1, x2, y2 = _xyxy_from_yolo(box, w, h)
        nx1 = x1 * scale + tx
        ny1 = y1 * scale + ty
        nx2 = x2 * scale + tx
        ny2 = y2 * scale + ty
        nx1 = max(0.0, min(float(w), nx1))
        ny1 = max(0.0, min(float(h), ny1))
        nx2 = max(0.0, min(float(w), nx2))
        ny2 = max(0.0, min(float(h), ny2))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        new_box = _yolo_from_xyxy(cls, nx1, ny1, nx2, ny2, w, h)
        if new_box[3] <= 0.0 or new_box[4] <= 0.0:
            continue
        new_labels.append(new_box)
    return out, new_labels


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
    # Some Pillow builds do not support GaussianBlur on mode "F" reliably.
    # Use uint8 path for compatibility, then map back to float32 [0, 1].
    u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    blurred = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    return _clip01(np.array(blurred, dtype=np.float32) / 255.0)


def _random_cutout(
    img: np.ndarray, labels: Sequence[YoloLabel], rng: random.Random
) -> Tuple[np.ndarray, List[YoloLabel]]:
    h, w = img.shape
    out = img.copy()
    cutouts: List[Tuple[float, float, float, float]] = []
    num_holes = rng.randint(1, 3)
    for _ in range(num_holes):
        cw = max(2, int(round(rng.uniform(0.06, 0.2) * w)))
        ch = max(2, int(round(rng.uniform(0.06, 0.2) * h)))
        x0 = rng.randint(0, max(0, w - cw))
        y0 = rng.randint(0, max(0, h - ch))
        x1 = x0 + cw
        y1 = y0 + ch
        fill = float(rng.uniform(0.0, 1.0))
        out[y0:y1, x0:x1] = fill
        cutouts.append((x0, y0, x1, y1))

    new_labels: List[YoloLabel] = []
    for box in labels:
        cls, bx1, by1, bx2, by2 = _xyxy_from_yolo(box, w, h)
        box_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
        covered = 0.0
        for cx1, cy1, cx2, cy2 in cutouts:
            ix1 = max(bx1, cx1)
            iy1 = max(by1, cy1)
            ix2 = min(bx2, cx2)
            iy2 = min(by2, cy2)
            if ix2 > ix1 and iy2 > iy1:
                covered += (ix2 - ix1) * (iy2 - iy1)
        visible_ratio = max(0.0, 1.0 - covered / box_area)
        if visible_ratio < MIN_CUTOUT_VISIBLE_RATIO:
            continue
        new_labels.append(box)
    return _clip01(out), new_labels


def _random_clahe_like(img: np.ndarray, rng: random.Random) -> np.ndarray:
    # CLAHE is histogram/LUT-based; uint8 binning keeps the implementation dependency-free and stable.
    u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    h, w = u8.shape
    tiles_x = rng.randint(6, 10)
    tiles_y = rng.randint(6, 10)
    clip_scale = rng.uniform(1.5, 3.0)
    out = np.empty_like(u8)

    for ty in range(tiles_y):
        y0 = (ty * h) // tiles_y
        y1 = ((ty + 1) * h) // tiles_y
        for tx in range(tiles_x):
            x0 = (tx * w) // tiles_x
            x1 = ((tx + 1) * w) // tiles_x
            tile = u8[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            hist = np.bincount(tile.reshape(-1), minlength=256).astype(np.float64)
            clip_limit = max(1.0, clip_scale * tile.size / 256.0)
            excess = np.maximum(hist - clip_limit, 0.0).sum()
            hist = np.minimum(hist, clip_limit)
            hist += excess / 256.0
            cdf = np.cumsum(hist)
            if cdf[-1] <= 0:
                out[y0:y1, x0:x1] = tile
                continue
            lut = np.floor(cdf * 255.0 / cdf[-1]).astype(np.uint8)
            out[y0:y1, x0:x1] = lut[tile]
    return _clip01(out.astype(np.float32) / 255.0)


def _random_histogram_perturb(img: np.ndarray, rng: random.Random) -> np.ndarray:
    low_q = rng.uniform(0.0, 4.0)
    high_q = rng.uniform(96.0, 100.0)
    low = float(np.percentile(img, low_q))
    high = float(np.percentile(img, high_q))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return img
    stretched = np.clip((img - low) / (high - low), 0.0, 1.0)
    s = rng.uniform(0.85, 1.15)
    o = rng.uniform(-0.08, 0.08)
    return _clip01(stretched * s + o)


def _random_blur_noise_combo(img: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    out = _random_blur(img, rng)
    sigma = rng.uniform(0.005, 0.025)
    noise = np_rng.normal(0.0, sigma, size=img.shape)
    return _clip01(out + noise)


def _compose_mosaic_from_samples(
    current_img: np.ndarray,
    chosen: Sequence[Tuple[np.ndarray, Sequence[YoloLabel]]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[YoloLabel]]:
    h, w = current_img.shape
    cut_x = rng.randint(max(1, int(0.35 * w)), max(1, int(0.65 * w)))
    cut_y = rng.randint(max(1, int(0.35 * h)), max(1, int(0.65 * h)))

    patches = [
        (0, 0, cut_x, cut_y),
        (cut_x, 0, w - cut_x, cut_y),
        (0, cut_y, cut_x, h - cut_y),
        (cut_x, cut_y, w - cut_x, h - cut_y),
    ]

    mosaic = np.zeros((h, w), dtype=np.float32)
    out_labels: List[YoloLabel] = []
    for (dst_x, dst_y, dst_w, dst_h), (src_img, src_labels) in zip(patches, chosen):
        if dst_w <= 0 or dst_h <= 0:
            continue
        src_h, src_w = src_img.shape
        resized = np.array(
            Image.fromarray(src_img, mode="F").resize((dst_w, dst_h), resample=Image.BILINEAR),
            dtype=np.float32,
        )
        mosaic[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w] = resized

        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        for box in src_labels:
            cls, x1, y1, x2, y2 = _xyxy_from_yolo(box, src_w, src_h)
            nx1 = dst_x + x1 * scale_x
            ny1 = dst_y + y1 * scale_y
            nx2 = dst_x + x2 * scale_x
            ny2 = dst_y + y2 * scale_y
            nx1 = max(0.0, min(float(w), nx1))
            ny1 = max(0.0, min(float(h), ny1))
            nx2 = max(0.0, min(float(w), nx2))
            ny2 = max(0.0, min(float(h), ny2))
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            out_labels.append(_yolo_from_xyxy(cls, nx1, ny1, nx2, ny2, w, h))
    return _clip01(mosaic), out_labels


def _random_mosaic(
    current_img: np.ndarray,
    current_labels: Sequence[YoloLabel],
    sample_pool: Sequence[Tuple[np.ndarray, Sequence[YoloLabel]]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[YoloLabel]]:
    chosen: List[Tuple[np.ndarray, Sequence[YoloLabel]]] = [(current_img, current_labels)]
    for _ in range(3):
        chosen.append(sample_pool[rng.randrange(len(sample_pool))])
    return _compose_mosaic_from_samples(current_img, chosen, rng)


def _random_mosaic_with_sampler(
    current_img: np.ndarray,
    current_labels: Sequence[YoloLabel],
    sample_count: int,
    sample_getter: Callable[[int], Tuple[np.ndarray, Sequence[YoloLabel]]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[YoloLabel]]:
    chosen: List[Tuple[np.ndarray, Sequence[YoloLabel]]] = [(current_img, current_labels)]
    for _ in range(3):
        chosen.append(sample_getter(rng.randrange(sample_count)))
    return _compose_mosaic_from_samples(current_img, chosen, rng)


def _apply_random_augmentations(
    img_f32: np.ndarray,
    labels: Sequence[YoloLabel],
    rng: random.Random,
    np_rng: np.random.Generator,
    sample_pool: Sequence[Tuple[np.ndarray, Sequence[YoloLabel]]] | None = None,
    sample_count: int = 0,
    sample_getter: Callable[[int], Tuple[np.ndarray, Sequence[YoloLabel]]] | None = None,
    aug_probs: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, List[YoloLabel]]:
    img = img_f32.copy()
    aug_labels = list(labels)
    probs = aug_probs or {}

    if rng.random() < probs.get("mosaic", 0.35):
        if sample_pool:
            img, aug_labels = _random_mosaic(img, aug_labels, sample_pool, rng)
        elif sample_getter is not None and sample_count > 0:
            img, aug_labels = _random_mosaic_with_sampler(img, aug_labels, sample_count, sample_getter, rng)
    if rng.random() < 0.6:
        img, aug_labels = _random_crop_and_resize(img, aug_labels, rng)
    if rng.random() < 0.6:
        img, aug_labels = _random_rotate_90(img, aug_labels, rng)
    if rng.random() < 0.5:
        img, aug_labels = _random_flip(img, aug_labels, rng)
    if rng.random() < probs.get("translate_scale", 0.6):
        img, aug_labels = _random_translate_scale(img, aug_labels, rng)
    if rng.random() < probs.get("cutout", 0.45):
        img, aug_labels = _random_cutout(img, aug_labels, rng)

    if rng.random() < 0.7:
        img = _random_contrast(img, rng)
    if rng.random() < 0.7:
        img = _random_brightness(img, rng)
    if rng.random() < probs.get("clahe", 0.45):
        img = _random_clahe_like(img, rng)
    if rng.random() < probs.get("gamma", 0.5):
        img = _random_gamma(img, rng)
    if rng.random() < probs.get("hist_perturb", 0.5):
        img = _random_histogram_perturb(img, rng)
    if rng.random() < 0.4:
        img = _random_gaussian_noise(img, rng, np_rng)
    if rng.random() < 0.4:
        img = _random_blur(img, rng)
    if rng.random() < probs.get("blur_noise_combo", 0.35):
        img = _random_blur_noise_combo(img, rng, np_rng)

    return _clip01(img), aug_labels


def _check_uint16_augmentation_compatibility(
    seed: int = 42,
    aug_probs: Dict[str, float] | None = None,
    normalize_mode: str = "per_image",
) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sample = np.linspace(0, 65535, UINT16_CHECK_IMAGE_SIZE * UINT16_CHECK_IMAGE_SIZE).reshape(
        UINT16_CHECK_IMAGE_SIZE, UINT16_CHECK_IMAGE_SIZE
    ).astype(np.uint16)
    base_img = _convert_to_float32_single_channel(sample, normalize_mode=normalize_mode)
    labels: List[YoloLabel] = [(0, 0.5, 0.5, 0.4, 0.4)]

    sample_pool = [(base_img, labels)]
    checks = [
        ("mosaic", lambda: _random_mosaic(base_img, labels, sample_pool, rng)),
        ("random_crop", lambda: _random_crop_and_resize(base_img, labels, rng)),
        ("random_rotation", lambda: _random_rotate_90(base_img, labels, rng)),
        ("flip", lambda: _random_flip(base_img, labels, rng)),
        ("translate_scale", lambda: _random_translate_scale(base_img, labels, rng)),
        ("cutout", lambda: _random_cutout(base_img, labels, rng)),
        ("random_contrast", lambda: (_random_contrast(base_img, rng), labels)),
        ("random_brightness", lambda: (_random_brightness(base_img, rng), labels)),
        ("clahe_like", lambda: (_random_clahe_like(base_img, rng), labels)),
        ("random_gamma", lambda: (_random_gamma(base_img, rng), labels)),
        ("histogram_perturb", lambda: (_random_histogram_perturb(base_img, rng), labels)),
        ("gaussian_noise", lambda: (_random_gaussian_noise(base_img, rng, np_rng), labels)),
        ("gaussian_blur", lambda: (_random_blur(base_img, rng), labels)),
        ("blur_noise_combo", lambda: (_random_blur_noise_combo(base_img, rng, np_rng), labels)),
        (
            "combined_pipeline",
            lambda: _apply_random_augmentations(base_img, labels, rng, np_rng, sample_pool, aug_probs=aug_probs),
        ),
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


def _is_yolo_dataset_layout(dataset_root: Path) -> bool:
    required_dirs = (
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    )
    return all(p.exists() and p.is_dir() for p in required_dirs)


def _list_tiff_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    )


def _ensure_empty_yolo_labels(yolo_root: Path) -> int:
    created = 0
    for split in ("train", "val"):
        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        if not image_dir.exists():
            continue
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in _list_tiff_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                continue
            label_path.write_text("", encoding="utf-8")
            created += 1
    return created


def _is_flat_raw_layout(dataset_root: Path) -> bool:
    images_dir = dataset_root / "images"
    ann_dir = dataset_root / "yolo_annotations"
    if not (images_dir.exists() and images_dir.is_dir() and ann_dir.exists() and ann_dir.is_dir()):
        return False
    if _is_yolo_dataset_layout(dataset_root):
        return False
    return len(_list_tiff_images(images_dir)) > 0


def _prepare_dataset_layout(
    dataset_root: Path,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Path:
    """Ensure dataset is in YOLO layout; convert flat raw layout if needed."""
    if _is_yolo_dataset_layout(dataset_root):
        return dataset_root

    if not _is_flat_raw_layout(dataset_root):
        raise RuntimeError(
            "Unsupported dataset layout. Expected either YOLO layout "
            "(images/train,val + labels/train,val) or flat raw layout (images + yolo_annotations)."
        )

    if not (0.0 < float(split_ratio) < 1.0):
        raise ValueError(f"--raw-split-ratio must be in (0, 1), got {split_ratio}")

    converted_root = dataset_root.parent / f"{dataset_root.name}_yolo"
    if converted_root.exists():
        return converted_root

    images_dir = dataset_root / "images"
    ann_dir = dataset_root / "yolo_annotations"
    image_paths = _list_tiff_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No TIFF images found in raw images directory: {images_dir}")

    paired: List[Tuple[Path, Path | None]] = []
    missing_label_count = 0
    for image_path in image_paths:
        label_path = ann_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            paired.append((image_path, None))
            missing_label_count += 1
            continue
        paired.append((image_path, label_path))

    rng = random.Random(split_seed)
    rng.shuffle(paired)
    split_idx = int(round(len(paired) * split_ratio))
    split_idx = max(1, min(len(paired) - 1, split_idx)) if len(paired) > 1 else 1
    train_pairs = paired[:split_idx]
    val_pairs = paired[split_idx:] if len(paired) > 1 else paired

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (converted_root / sub).mkdir(parents=True, exist_ok=True)

    for image_path, label_path in train_pairs:
        shutil.copy2(image_path, converted_root / "images" / "train" / image_path.name)
        dst_label = converted_root / "labels" / "train" / f"{image_path.stem}.txt"
        if label_path is None:
            dst_label.write_text("", encoding="utf-8")
        else:
            shutil.copy2(label_path, dst_label)
    for image_path, label_path in val_pairs:
        shutil.copy2(image_path, converted_root / "images" / "val" / image_path.name)
        dst_label = converted_root / "labels" / "val" / f"{image_path.stem}.txt"
        if label_path is None:
            dst_label.write_text("", encoding="utf-8")
        else:
            shutil.copy2(label_path, dst_label)

    print(f"Converted raw dataset layout to YOLO split: {converted_root}")
    print(f"Train/Val split: {len(train_pairs)}/{len(val_pairs)} (seed={split_seed}, ratio={split_ratio})")
    if missing_label_count > 0:
        print(
            f"Raw dataset contains {missing_label_count} unlabeled images. "
            "Created empty YOLO label files for them."
        )
    return converted_root


def convert_tifs_to_float32(
    dataset_root: Path,
    reuse_prepared: bool = False,
    force_rebuild_prepared: bool = False,
    augment_copies: int = 0,
    augment_seed: int = 42,
    normalize_mode: str = "per_image",
    aug_probs: Dict[str, float] | None = None,
) -> Path:
    prepared_root = dataset_root.parent / f"{dataset_root.name}_prepared"
    if reuse_prepared and force_rebuild_prepared:
        raise ValueError("--reuse-prepared and --force-rebuild-prepared cannot be used together.")
    if prepared_root.exists():
        if reuse_prepared:
            created = _ensure_empty_yolo_labels(prepared_root)
            if created > 0:
                print(f"Prepared dataset missing {created} label files; created empty labels automatically.")
            return prepared_root
        if force_rebuild_prepared:
            try:
                shutil.rmtree(prepared_root)
            except OSError as exc:
                raise RuntimeError(f"Failed to clean prepared dataset directory: {prepared_root}") from exc
        else:
            created = _ensure_empty_yolo_labels(prepared_root)
            if created > 0:
                print(f"Prepared dataset missing {created} label files; created empty labels automatically.")
            return prepared_root
    shutil.copytree(dataset_root, prepared_root)
    created = _ensure_empty_yolo_labels(prepared_root)
    if created > 0:
        print(f"Prepared dataset missing {created} label files; created empty labels automatically.")

    rng = random.Random(augment_seed)
    np_rng = np.random.default_rng(augment_seed)
    for split in ("train", "val"):
        image_dir = prepared_root / "images" / split
        label_dir = prepared_root / "labels" / split
        if not image_dir.exists():
            continue
        all_paths = _list_tiff_images(image_dir)
        for tif_path in all_paths:
            # Read from bytes to avoid Windows file-locking issues and preserve source bit-depth.
            arr = _read_tiff_array_lossless(tif_path)
            arr_f32 = _convert_to_float32_single_channel(arr, normalize_mode=normalize_mode)
            save_path = tif_path.with_suffix(".tif")
            Image.fromarray(arr_f32, mode="F").save(save_path)
            if tif_path != save_path:
                tif_path.unlink()
        if split == "train" and augment_copies > 0:
            # All train images are normalized and rewritten to .tif above, so we augment from .tif only.
            train_base_paths = sorted(image_dir.glob("*.tif"))
            if train_base_paths:
                sample_cache: OrderedDict[Path, Tuple[np.ndarray, List[YoloLabel]]] = OrderedDict()

                def _load_train_sample(path: Path) -> Tuple[np.ndarray, List[YoloLabel]]:
                    cached = sample_cache.get(path)
                    if cached is not None:
                        sample_cache.move_to_end(path)
                        return cached
                    arr = _read_tiff_array_lossless(path).astype(np.float32)
                    arr = _clip01(arr)
                    labels = _read_yolo_labels(label_dir / f"{path.stem}.txt")
                    sample = (arr, labels)
                    sample_cache[path] = sample
                    if len(sample_cache) > MAX_AUG_IMAGE_CACHE:
                        sample_cache.popitem(last=False)
                    return sample

                sample_count = len(train_base_paths)

                def _sample_getter(index: int) -> Tuple[np.ndarray, Sequence[YoloLabel]]:
                    return _load_train_sample(train_base_paths[index])

                for base_path in train_base_paths:
                    base_arr, base_labels = _load_train_sample(base_path)
                    for idx in range(augment_copies):
                        aug_img, aug_labels = _apply_random_augmentations(
                            base_arr,
                            base_labels,
                            rng,
                            np_rng,
                            sample_count=sample_count,
                            sample_getter=_sample_getter,
                            aug_probs=aug_probs,
                        )
                        aug_stem = f"{base_path.stem}_aug{idx + 1}"
                        Image.fromarray(aug_img, mode="F").save(image_dir / f"{aug_stem}.tif")
                        _write_yolo_labels(label_dir / f"{aug_stem}.txt", aug_labels)
    return prepared_root


def write_data_yaml(dataset_root: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "channels": 1,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    data_yaml = dataset_root / "rtdetr_data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: {}\nval: {}\nchannels: {}\nnames:\n{}\n".format(
            data["path"],
            data["train"],
            data["val"],
            data["channels"],
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


def _resolve_run_dir(train_result, model, default_run_dir: Path) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None) if trainer is not None else None
    return Path(save_dir) if save_dir else default_run_dir


def _enable_single_channel_input_compat(model_like) -> None:
    """Allow 1-channel TIFF inputs with RGB-pretrained Ultralytics models."""
    pt_model = getattr(model_like, "model", model_like)
    if hasattr(pt_model, "module") and isinstance(pt_model.module, torch.nn.Module):
        pt_model = pt_model.module
    if not isinstance(pt_model, torch.nn.Module):
        return
    if getattr(pt_model, "_single_channel_compat_enabled", False):
        return

    first_conv_name = ""
    first_conv = None
    for name, module in pt_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv_name = name
            first_conv = module
            break

    if first_conv is None or int(first_conv.in_channels) != 3 or int(first_conv.groups) != 1:
        return

    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=1,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
        device=first_conv.weight.device,
        dtype=first_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    parent = pt_model
    parts = first_conv_name.split(".") if first_conv_name else []
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    if parts:
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = new_conv
        else:
            setattr(parent, leaf, new_conv)

    pt_model._single_channel_compat_enabled = True


def main() -> None:
    args = parse_args()
    if args.augment_copies < 0:
        raise ValueError("--augment-copies must be >= 0")
    aug_probs = {
        "mosaic": args.augment_mosaic_prob,
        "translate_scale": args.augment_translate_scale_prob,
        "cutout": args.augment_cutout_prob,
        "clahe": args.augment_clahe_prob,
        "gamma": args.augment_gamma_prob,
        "hist_perturb": args.augment_hist_perturb_prob,
        "blur_noise_combo": args.augment_blur_noise_combo_prob,
    }
    for key, val in aug_probs.items():
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"Augmentation probability '{key}' must be in [0, 1], got {val}.")

    _check_uint16_augmentation_compatibility(
        seed=args.augment_seed,
        aug_probs=aug_probs,
        normalize_mode=args.normalize_mode,
    )
    print("uint16 TIFF augmentation compatibility check passed.")

    prepared_root = convert_tifs_to_float32(
        _prepare_dataset_layout(
            args.dataset_root,
            split_ratio=args.raw_split_ratio,
            split_seed=args.raw_split_seed,
        ),
        reuse_prepared=args.reuse_prepared,
        force_rebuild_prepared=args.force_rebuild_prepared,
        augment_copies=args.augment_copies,
        augment_seed=args.augment_seed,
        normalize_mode=args.normalize_mode,
        aug_probs=aug_probs,
    )
    data_yaml = write_data_yaml(prepared_root, args.class_names)
    model_path, model_family = resolve_model_path(
        args.model,
        args.weights_dir,
        preset_sha256=args.preset_sha256,
    )
    model = RTDETR(model_path) if model_family == "rtdetr" else YOLO(model_path)
    _enable_single_channel_input_compat(model)
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
            if isinstance(value, numbers.Real):
                writer.add_scalar(str(key), float(value), epoch)
        val_loss = _val_loss_total(metrics)
        if np.isfinite(val_loss):
            writer.add_scalar("val/loss_total", val_loss, epoch)

    def on_train_end(trainer) -> None:
        _close_tb_writer()

    def on_train_start(trainer) -> None:
        _enable_single_channel_input_compat(trainer.model)
        ema_model = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema_model is not None:
            _enable_single_channel_input_compat(ema_model)

    model.add_callback("on_train_start", on_train_start)
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
