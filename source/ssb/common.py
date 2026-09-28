from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-12)


def rect_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("invalid positive log-uniform range")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_side(rng: np.random.Generator, low: int, high: int) -> int:
    if low > high:
        raise ValueError("low > high")
    if low == high:
        return int(low)
    return int(round(sample_log_uniform(rng, float(low), float(high))))


def centered_patch_box(box_xyxy: Sequence[float], side: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    px1 = int(round(cx - side / 2.0))
    py1 = int(round(cy - side / 2.0))
    return px1, py1, px1 + side, py1 + side


def fit_patch_box(x1: int, y1: int, side: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    x2, y2 = x1 + side, y1 + side
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None
    return x1, y1, x2, y2


def find_background_patch_boxes(
    width: int,
    height: int,
    forbidden_boxes: Sequence[Sequence[float]],
    sides: Sequence[int],
    rng: np.random.Generator,
    max_trials: int = 1000,
) -> list[tuple[int, int, int, int]]:
    chosen: list[tuple[int, int, int, int]] = []
    all_forbidden = [tuple(map(float, b)) for b in forbidden_boxes]
    for side in sides:
        if side >= width or side >= height:
            continue
        found = None
        for _ in range(max_trials):
            x1 = int(rng.integers(0, width - side + 1))
            y1 = int(rng.integers(0, height - side + 1))
            cand = (x1, y1, x1 + side, y1 + side)
            if any(rect_intersects(cand, b) for b in all_forbidden):
                continue
            if any(rect_intersects(cand, b) for b in chosen):
                continue
            found = cand
            break
        if found is not None:
            chosen.append(found)
    return chosen


def paste_patch_numpy_bgr(
    image_bgr: np.ndarray,
    patch_bgr_01: np.ndarray,
    box: Sequence[int],
    blend_ratio: float = 1.0,
) -> np.ndarray:
    if not (0.0 <= blend_ratio <= 1.0):
        raise ValueError("blend_ratio must be in [0,1]")
    x1, y1, x2, y2 = map(int, box)
    out = image_bgr.copy()
    region = out[y1:y2, x1:x2].astype(np.float32) / 255.0
    patch = patch_bgr_01.astype(np.float32)
    if region.shape != patch.shape:
        raise ValueError(f"region shape {region.shape} != patch shape {patch.shape}")
    mixed = (1.0 - blend_ratio) * region + blend_ratio * patch
    out[y1:y2, x1:x2] = np.clip(np.round(mixed * 255.0), 0, 255).astype(np.uint8)
    return out


def paste_patch_tensor_rgb(
    image_rgb_01: Tensor,
    patch_rgb_01: Tensor,
    box: Sequence[int],
    blend_ratio: float = 1.0,
) -> Tensor:
    if not (0.0 <= blend_ratio <= 1.0):
        raise ValueError("blend_ratio must be in [0,1]")
    x1, y1, x2, y2 = map(int, box)
    out = image_rgb_01.clone()
    region = out[:, y1:y2, x1:x2]
    if tuple(region.shape) != tuple(patch_rgb_01.shape):
        raise ValueError(f"region shape {tuple(region.shape)} != patch {tuple(patch_rgb_01.shape)}")
    out[:, y1:y2, x1:x2] = (1.0 - blend_ratio) * region + blend_ratio * patch_rgb_01
    return out


def yolo_label_line(class_index: int, box_xywh: Sequence[float], width: int, height: int) -> str:
    x, y, w, h = map(float, box_xywh)
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    return f"{class_index} {xc:.8f} {yc:.8f} {w/width:.8f} {h/height:.8f}\n"
