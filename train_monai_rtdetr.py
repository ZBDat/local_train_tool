#!/usr/bin/env python3
"""Train RT-DETR on 16-bit TIFF data with a MONAI-backed data pipeline.

Unlike ``train_rtdetr.py``, this pipeline keeps the full bit depth of uint16
TIFF inputs.  Images are loaded with MONAI (ITK reader), normalized to
float32 in [0, 1] and fed to the RT-DETR transformer detector with a custom
training loop.  It intentionally bypasses the Ultralytics training pipeline,
which hard-codes 8-bit uint8 assumptions (mosaic canvas, ``/255`` rescaling)
and therefore corrupts float32/[0,1] training data.

The RT-DETR model (forward/loss/weights) is reused from Ultralytics
(AGPL-3.0, same as the rest of this project).
"""
import argparse
import csv
import json
import random
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import train_rtdetr as tr

try:
    from monai.transforms import LoadImage
    MONAI_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    MONAI_AVAILABLE = False

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import plot_images


YoloLabel = Tuple[int, float, float, float, float]
MAX_AUG_IMAGE_CACHE = 16
DEFAULT_PAD_VAL = 0.447  # 114 / 255, matches COCO-pretraining letterbox fill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RT-DETR on 16-bit TIFF data with a MONAI 16-bit data pipeline."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="YOLO dataset root directory.")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names, e.g. --class-names object")
    parser.add_argument(
        "--model",
        type=str,
        default="weights/rtdetr-l.pt",
        help="RT-DETR checkpoint path or preset key (e.g. coco-rtdetr-l / coco-rtdetr-x).",
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"), help="Directory for preset weights.")
    parser.add_argument("--preset-sha256", type=str, default="", help="Optional expected SHA256 for URL-based presets.")
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=tr.NORMALIZE_MODES,
        default="per_image",
        help="Image normalization mode: per_image / fixed_6bit / fixed_uint16.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="monai_rtdetr_train")
    parser.add_argument("--device", type=str, default="", help="cuda device, e.g. '0' or 'cpu'.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 = in main process).")
    parser.add_argument(
        "--pad-val", type=float, default=DEFAULT_PAD_VAL, help="Letterbox padding value in [0,1] (default 0.447)."
    )
    parser.add_argument(
        "--val-metric", type=str, choices=("map", "loss"), default="map", help="Metric used to select the best model."
    )
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation scoring.")
    parser.add_argument("--resume", type=str, default="", help="Path to a last.pt checkpoint to resume from.")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP when training on CUDA.")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP.")
    parser.add_argument(
        "--plot-batch", action="store_true", default=True,
        help="Save train/val batch preview grids (train_batch_preview.jpg) before training starts.",
    )
    parser.add_argument("--no-plot-batch", action="store_false", dest="plot_batch", help="Disable batch preview.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-mosaic-prob", type=float, default=0.35)
    parser.add_argument("--augment-translate-scale-prob", type=float, default=0.6)
    parser.add_argument("--augment-cutout-prob", type=float, default=0.45)
    parser.add_argument("--augment-clahe-prob", type=float, default=0.45)
    parser.add_argument("--augment-gamma-prob", type=float, default=0.5)
    parser.add_argument("--augment-hist-perturb-prob", type=float, default=0.5)
    parser.add_argument("--augment-blur-noise-combo-prob", type=float, default=0.35)
    parser.add_argument("--raw-split-ratio", type=float, default=0.8)
    parser.add_argument("--raw-split-seed", type=int, default=42)
    return parser.parse_args()


def _build_aug_probs(args: argparse.Namespace) -> Dict[str, float]:
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
    return aug_probs


def _worker_init_fn(worker_id: int) -> None:
    """Top-level worker seed function (must be picklable on Windows)."""
    np.random.seed(worker_id)


# ---------------------------------------------------------------------------
# 16-bit image loading (MONAI-first, PIL fallback)
# ---------------------------------------------------------------------------
def _load_image_monai(path: Path) -> np.ndarray:
    """Load a TIFF preserving its native bit depth (uint16 stays uint16).

    MONAI 1.6.x returns 2D grayscale images in (W, H) order (channel-last
    convention); transpose back to the standard (H, W) layout to keep the
    same orientation as the source TIFF.
    """
    if MONAI_AVAILABLE:
        for reader in ("itkreader", "pilreader"):
            try:
                arr = np.asarray(LoadImage(reader=reader, image_only=True, ensure_channel_first=False)(str(path)))
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    arr = arr.T
                return arr
            except Exception as exc:  # noqa: BLE001 - try next reader
                warnings.warn(f"MONAI reader '{reader}' failed for {path}: {exc}", RuntimeWarning, stacklevel=2)
    return tr._read_tiff_array_lossless(path)


def _to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (1, H, W) or (H, W, 1) -> (H, W)
        if 1 in arr.shape:
            axis = 0 if arr.shape[0] == 1 else -1
            return np.squeeze(arr, axis=axis)
        return arr.mean(axis=2)
    raise TypeError(f"Unsupported array shape after load: {arr.shape}")


# ---------------------------------------------------------------------------
# Letterbox (keep aspect ratio, pad with pad_val)
# ---------------------------------------------------------------------------
def _letterbox(img: np.ndarray, imgsz: int, pad_val: float) -> Tuple[np.ndarray, Tuple[float, int, int]]:
    h, w = img.shape
    r = min(imgsz / w, imgsz / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    resized = np.array(
        Image.fromarray(img, mode="F").resize((nw, nh), resample=Image.BILINEAR), dtype=np.float32
    )
    canvas = np.full((imgsz, imgsz), float(pad_val), dtype=np.float32)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dh : dh + nh, dw : dw + nw] = resized
    return canvas, (r, dw, dh)


def _boxes_to_normalized_xywh(
    labels: Sequence[YoloLabel], img_h: int, img_w: int, r: float, dw: int, dh: int, imgsz: int, nc: int
) -> List[Tuple[int, float, float, float, float]]:
    """Map augmented (normalized) boxes into the letterboxed imgsz image and normalize to [0,1] xywh.

    RT-DETR's loss expects GT boxes in normalized xywh format in [0,1]
    (the decoder outputs normalized xywh; the loss/matcher/denoising all
    operate in that space).
    """
    out: List[Tuple[int, float, float, float, float]] = []
    for cls, x, y, w, h in labels:
        if cls < 0 or cls >= nc:
            continue
        x1 = (x - w / 2.0) * img_w * r + dw
        y1 = (y - h / 2.0) * img_h * r + dh
        x2 = (x + w / 2.0) * img_w * r + dw
        y2 = (y + h / 2.0) * img_h * r + dh
        x1 = max(0.0, min(float(imgsz), x1))
        y1 = max(0.0, min(float(imgsz), y1))
        x2 = max(0.0, min(float(imgsz), x2))
        y2 = max(0.0, min(float(imgsz), y2))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1e-6 or bh <= 1e-6:
            continue
        cx = (x1 + x2) / 2.0 / imgsz
        cy = (y1 + y2) / 2.0 / imgsz
        bw /= imgsz
        bh /= imgsz
        out.append((cls, cx, cy, bw, bh))
    return out


# ---------------------------------------------------------------------------
# MONAI-backed dataset
# ---------------------------------------------------------------------------
class MonaiRTDETRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        imgsz: int,
        nc: int,
        mode: str = "train",
        normalize_mode: str = "per_image",
        seed: int = 42,
        aug_probs: Dict[str, float] | None = None,
        pad_val: float = DEFAULT_PAD_VAL,
    ) -> None:
        self.image_paths = tr._list_tiff_images(image_dir)
        self.label_dir = label_dir
        self.imgsz = imgsz
        self.nc = nc
        self.mode = mode
        self.normalize_mode = normalize_mode
        self.aug_probs = aug_probs or {}
        self.pad_val = pad_val
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._cache: "OrderedDict[int, Tuple[np.ndarray, List[YoloLabel]]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_raw_sample(self, index: int) -> Tuple[np.ndarray, List[YoloLabel]]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        path = self.image_paths[index]
        arr = _to_2d(_load_image_monai(path))
        img = tr._convert_to_float32_single_channel(arr, normalize_mode=self.normalize_mode)
        labels = tr._read_yolo_labels(self.label_dir / f"{path.stem}.txt")
        sample = (img, labels)
        self._cache[index] = sample
        if len(self._cache) > MAX_AUG_IMAGE_CACHE:
            self._cache.popitem(last=False)
        return sample

    def _sample_getter(self, index: int) -> Tuple[np.ndarray, Sequence[YoloLabel]]:
        return self._load_raw_sample(index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, labels = self._load_raw_sample(index)
        ori_shape = (int(img.shape[0]), int(img.shape[1]))
        if self.mode == "train":
            img, labels = tr._apply_random_augmentations(
                img,
                labels,
                self.rng,
                self.np_rng,
                sample_count=len(self),
                sample_getter=self._sample_getter,
                aug_probs=self.aug_probs,
            )
        img_pad, (r, dw, dh) = _letterbox(img, self.imgsz, self.pad_val)
        boxes = _boxes_to_normalized_xywh(labels, img.shape[0], img.shape[1], r, dw, dh, self.imgsz, self.nc)

        cls = torch.tensor([b[0] for b in boxes], dtype=torch.long)
        bboxes = torch.tensor([b[1:] for b in boxes], dtype=torch.float32).reshape(-1, 4)
        batch_idx = torch.zeros(cls.shape[0], dtype=torch.long)
        img_t = torch.from_numpy(img_pad[None]).contiguous()  # (1, H, W) float32 [0,1]
        return {
            "img": img_t,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "im_file": str(self.image_paths[index]),
            "ori_shape": ori_shape,
        }


def _collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    img = torch.stack([b["img"] for b in batch], dim=0)
    cls = torch.cat([b["cls"] for b in batch], dim=0)
    bboxes = torch.cat([b["bboxes"] for b in batch], dim=0)
    batch_idx = torch.cat([b["batch_idx"] + i for i, b in enumerate(batch)], dim=0)
    return {
        "img": img,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": [b["im_file"] for b in batch],
        "ori_shape": [b["ori_shape"] for b in batch],
        "resized_shape": [tuple(b["img"].shape[1:]) for b in batch],
        "ratio_pad": [(1.0, 1.0) for _ in batch],
    }


# ---------------------------------------------------------------------------
# RT-DETR inference post-processing (mirrors RTDETRValidator.postprocess)
# ---------------------------------------------------------------------------
def _postprocess_rtdetr(preds: Any, imgsz: int, conf: float) -> List[Dict[str, torch.Tensor]]:
    if not isinstance(preds, (list, tuple)):
        preds = [preds, None]
    bboxes, scores = preds[0].split((4, preds[0].shape[-1] - 4), dim=-1)
    bboxes *= imgsz
    outputs: List[Dict[str, torch.Tensor]] = []
    for bbox, score_row in zip(bboxes, scores):
        bbox = xywh2xyxy(bbox)
        score, cls = score_row.max(-1)
        pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
        keep = score > conf
        pred = pred[keep]
        pred = pred[pred[:, 4].argsort(descending=True)]
        outputs.append({"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5]})
    return outputs


def _match_predictions(
    pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, iouv: torch.Tensor
) -> torch.Tensor:
    """Greedy IoU matching replicating Ultralytics' validator logic."""
    pred_classes = pred_classes.cpu()
    true_classes = true_classes.cpu()
    iou = iou.cpu()
    correct = np.zeros((pred_classes.shape[0], iouv.numel()), dtype=bool)
    correct_class = (true_classes[:, None] == pred_classes).numpy()
    iou_np = (iou * torch.from_numpy(correct_class)).numpy()
    thresh = iouv.numpy()
    for i, t in enumerate(thresh):
        matches = np.nonzero(iou_np >= t)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.from_numpy(correct)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _build_model(model_path: Path, nc: int, device: torch.device) -> RTDETRDetectionModel:
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model = RTDETRDetectionModel(cfg=ckpt["model"].yaml, ch=3, nc=nc, verbose=False)
    model.load(ckpt)
    model.nc = nc
    tr._enable_single_channel_input_compat(model)
    return model.to(device)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _run_validation(
    model: RTDETRDetectionModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    imgsz: int,
    conf: float,
    nc: int,
    class_names: List[str],
) -> Tuple[Dict[str, float], float]:
    model.eval()
    metrics = DetMetrics(names={i: n for i, n in enumerate(class_names)})
    iouv = torch.linspace(0.5, 0.95, 10)
    total_loss = 0.0
    n_loss = 0
    with torch.no_grad():
        for batch in loader:
            if batch["cls"].shape[0] == 0:
                continue
            img = batch["img"].to(device, non_blocking=True)
            preds = model(img)
            loss_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            vl, _ = model.loss(loss_batch, preds)
            total_loss += float(vl.item()) * img.shape[0]
            n_loss += img.shape[0]

            preds_list = _postprocess_rtdetr(preds, imgsz, conf)
            bbox_t = batch["bboxes"].to(device)
            cls_t = batch["cls"].to(device).squeeze(-1)
            batch_idx = batch["batch_idx"].to(device)
            for si, pred in enumerate(preds_list):
                idx = batch_idx == si
                true_cls = cls_t[idx]
                # GT is stored normalized xywh; convert to pixel xyxy to match preds
                true_box = xywh2xyxy(bbox_t[idx]) * imgsz
                pred_box = pred["bboxes"]
                pred_conf = pred["conf"]
                pred_cls = pred["cls"]
                if pred_box.shape[0] == 0 or true_box.shape[0] == 0:
                    tp = torch.zeros((pred_box.shape[0], iouv.numel()), dtype=torch.bool)
                else:
                    iou = box_iou(true_box, pred_box)
                    tp = _match_predictions(pred_cls, true_cls, iou, iouv)
                metrics.update_stats(
                    {
                        "tp": tp.numpy(),
                        "conf": pred_conf.cpu().numpy(),
                        "pred_cls": pred_cls.cpu().numpy(),
                        "target_cls": true_cls.cpu().numpy(),
                        "target_img": np.unique(true_cls.cpu().numpy()),
                    }
                )
    if n_loss > 0:
        metrics.process(plot=False)
        results = metrics.results_dict
    else:
        results = {
            "metrics/precision(B)": 0.0,
            "metrics/recall(B)": 0.0,
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            "fitness": 0.0,
        }
    return results, total_loss / max(1, n_loss)


# ---------------------------------------------------------------------------
# Batch preview (like Ultralytics' train_batch*.jpg, generated before training)
# ---------------------------------------------------------------------------
def _plot_batch_preview(loader: torch.utils.data.DataLoader, fname: Path, names: Dict[int, str]) -> None:
    """Save a batch grid with GT boxes to disk (handles 1-channel float32 [0,1] images)."""
    batch = next(iter(loader))
    plot_images(
        labels=batch,
        paths=batch["im_file"],
        fname=str(fname),
        names=names,
        max_subplots=max(1, min(len(batch["im_file"]), 16)),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_epoch(
    model: RTDETRDetectionModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
) -> Tuple[float, float, float, float]:
    model.train()
    tot = [0.0, 0.0, 0.0]
    n = 0
    lr = optimizer.param_groups[0]["lr"]
    autocast = torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda")
    for batch in loader:
        if batch["cls"].shape[0] == 0:
            continue
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        with autocast:
            loss, items = model(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        tot[0] += float(items[0].item())
        tot[1] += float(items[1].item())
        tot[2] += float(items[2].item())
        n += 1
    scheduler.step()
    return tuple(t / max(1, n) for t in tot) + (lr,)


def _save_checkpoint(path: Path, model: RTDETRDetectionModel, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "model_yaml": model.yaml,
            "nc": model.nc,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    aug_probs = _build_aug_probs(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    if args.device:
        device = torch.device("cpu" if args.device.lower() == "cpu" else f"cuda:{args.device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model_path, model_family = tr.resolve_model_path(args.model, args.weights_dir, preset_sha256=args.preset_sha256)
    if model_family != "rtdetr":
        raise ValueError(f"train_monai_rtdetr.py only supports RT-DETR models, got family '{model_family}'.")
    model_path = Path(model_path)

    dataset_root = tr._prepare_dataset_layout(args.dataset_root, split_ratio=args.raw_split_ratio, split_seed=args.raw_split_seed)
    nc = len(args.class_names)
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"

    train_ds = MonaiRTDETRDataset(
        train_dir,
        dataset_root / "labels" / "train",
        imgsz=args.imgsz,
        nc=nc,
        mode="train",
        normalize_mode=args.normalize_mode,
        seed=args.seed,
        aug_probs=aug_probs,
        pad_val=args.pad_val,
    )
    val_ds = MonaiRTDETRDataset(
        val_dir,
        dataset_root / "labels" / "val",
        imgsz=args.imgsz,
        nc=nc,
        mode="val",
        normalize_mode=args.normalize_mode,
        seed=args.seed,
        pad_val=args.pad_val,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No TIFF images found in train dir: {train_dir}")
    print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=_collate_fn,
        pin_memory=device.type == "cuda", drop_last=False, worker_init_fn=_worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = _build_model(model_path, nc, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for _ in range(start_epoch):
        scheduler.step()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in vars(args).items()), encoding="utf-8"
    )
    writer = SummaryWriter(log_dir=str(run_dir))

    names = {i: n for i, n in enumerate(args.class_names)}
    if args.plot_batch:
        _plot_batch_preview(train_loader, run_dir / "train_batch_preview.jpg", names)
        print(f"Saved train batch preview: {run_dir / 'train_batch_preview.jpg'}", flush=True)
        if len(val_ds) > 0:
            _plot_batch_preview(val_loader, run_dir / "val_batch_preview.jpg", names)
            print(f"Saved val batch preview: {run_dir / 'val_batch_preview.jpg'}", flush=True)

    best_metric = -1.0
    results_csv_path = run_dir / "results.csv"
    with results_csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train/giou_loss", "train/cls_loss", "train/bbox_loss", "lr", "val/loss",
             "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        )

    print(f"Training RT-DETR on {len(train_ds)} samples, {nc} class(es), device={device}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        g, c, b, lr = _run_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, use_amp)
        writer.add_scalar("train/giou_loss", g, epoch)
        writer.add_scalar("train/cls_loss", c, epoch)
        writer.add_scalar("train/bbox_loss", b, epoch)
        writer.add_scalar("lr", lr, epoch)
        print(f"epoch {epoch + 1}/{args.epochs} | giou {g:.4f} cls {c:.4f} bbox {b:.4f} lr {lr:.2e} | {time.time() - t0:.1f}s", flush=True)

        val_results, val_loss = {}, 0.0
        if len(val_ds) > 0:
            val_results, val_loss = _run_validation(model, val_loader, device, args.imgsz, args.conf, nc, args.class_names)
            writer.add_scalar("val/loss", val_loss, epoch)
            for k, v in val_results.items():
                writer.add_scalar(k, float(v), epoch)
            map50 = float(val_results["metrics/mAP50(B)"])
            map50_95 = float(val_results["metrics/mAP50-95(B)"])
            print(f"  val  | loss {val_loss:.4f} mAP50 {map50:.4f} mAP50-95 {map50_95:.4f}", flush=True)
            metric = map50_95 if args.val_metric == "map" else -val_loss
            if metric > best_metric:
                best_metric = metric
                _save_checkpoint(run_dir / "best.pt", model, optimizer, epoch)
                (run_dir / "best_val_loss.json").write_text(
                    json.dumps({"best_epoch": epoch, "best_val_loss": val_loss, "metrics": val_results}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        _save_checkpoint(run_dir / "last.pt", model, optimizer, epoch)

        with results_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch + 1, g, c, b, lr, val_loss,
                 val_results.get("metrics/precision(B)", 0.0), val_results.get("metrics/recall(B)", 0.0),
                 val_results.get("metrics/mAP50(B)", 0.0), val_results.get("metrics/mAP50-95(B)", 0.0)]
            )

    writer.close()
    print(f"Training completed. Run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
