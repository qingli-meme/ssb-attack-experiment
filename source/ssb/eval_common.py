from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from source.ssb.common import (
    centered_patch_box,
    find_background_patch_boxes,
    iou_xyxy,
    paste_patch_numpy_bgr,
    rect_intersects,
    xywh_to_xyxy,
)
from source.ssb.trigger import ScaleSteerableTrigger


@dataclass
class Record:
    img_id: int
    path: Path
    width: int
    height: int
    boxes: np.ndarray      # [N,4] xyxy
    labels: np.ndarray     # [N] 0-based COCO80 index


@dataclass
class Prediction:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def load_records(coco_root: str | Path, max_images: int = -1) -> list[Record]:
    root = Path(coco_root)
    coco = COCO(str(root / "annotations" / "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    ids = sorted(coco.getImgIds())
    if max_images > 0:
        ids = ids[:max_images]

    records = []
    for img_id in ids:
        info = coco.loadImgs([img_id])[0]
        boxes, labels = [], []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None)):
            if ann.get("iscrowd", 0):
                continue
            boxes.append(xywh_to_xyxy(ann["bbox"]))
            labels.append(cat_to_idx[int(ann["category_id"])])
        records.append(
            Record(
                img_id=int(img_id),
                path=root / "images" / "val2017" / info["file_name"],
                width=int(info["width"]),
                height=int(info["height"]),
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64),
            )
        )
    return records


def clean_true_positive_gt_indices(record: Record, pred: Prediction, iou_thr: float = 0.5) -> list[int]:
    indices = []
    for gi, (gt_box, gt_label) in enumerate(zip(record.boxes, record.labels)):
        matched = False
        for pb, pl in zip(pred.boxes, pred.labels):
            if int(pl) == int(gt_label) and iou_xyxy(gt_box, pb) >= iou_thr:
                matched = True
                break
        if matched:
            indices.append(gi)
    return indices


def run_clean_predictions(records: Sequence[Record], predict: Callable[[np.ndarray], Prediction]) -> dict[int, Prediction]:
    cache = {}
    for rec in tqdm(records, desc="clean inference"):
        image = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(rec.path)
        cache[rec.img_id] = predict(image)
    return cache


def evaluate_oda(
    records: Sequence[Record],
    clean_predictions: Dict[int, Prediction],
    predict: Callable[[np.ndarray], Prediction],
    trigger: ScaleSteerableTrigger,
    trigger_sizes: Sequence[int],
    blend_ratio: float,
    iou_thr: float,
    bbox_to_trigger_ratio: float,
) -> list[dict]:
    results = []
    for side in trigger_sizes:
        patch = trigger.render_numpy_bgr(int(side))
        success = 0
        attempted = 0
        for rec in tqdm(records, desc=f"ODA side={side}"):
            tp_indices = clean_true_positive_gt_indices(rec, clean_predictions[rec.img_id], iou_thr)
            attacked = []
            image = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            poisoned = image
            for gi in tp_indices:
                gt = rec.boxes[gi]
                w = float(gt[2] - gt[0])
                h = float(gt[3] - gt[1])
                if min(w / side, h / side) < bbox_to_trigger_ratio:
                    continue
                pbox = centered_patch_box(gt, int(side))
                x1, y1, x2, y2 = pbox
                if x1 < 0 or y1 < 0 or x2 > rec.width or y2 > rec.height:
                    continue
                poisoned = paste_patch_numpy_bgr(poisoned, patch, pbox, blend_ratio)
                attacked.append(gi)
            if not attacked:
                continue
            post = predict(poisoned)
            for gi in attacked:
                gt = rec.boxes[gi]
                # Match the baseline evaluation: disappearance fails if any box
                # still overlaps the GT with IoU >= threshold, regardless of class.
                still_present = any(iou_xyxy(gt, pb) >= iou_thr for pb in post.boxes)
                success += int(not still_present)
                attempted += 1
        asr = success / attempted if attempted else float("nan")
        results.append({"scenario": "oda", "trigger_size": int(side), "asr": asr,
                        "success": success, "attempted": attempted})
    return results


def evaluate_oga(
    records: Sequence[Record],
    clean_predictions: Dict[int, Prediction],
    predict: Callable[[np.ndarray], Prediction],
    trigger: ScaleSteerableTrigger,
    trigger_sizes: Sequence[int],
    target_label: int,
    blend_ratio: float,
    seed: int,
) -> list[dict]:
    results = []
    for side in trigger_sizes:
        patch = trigger.render_numpy_bgr(int(side))
        success = 0
        attempted = 0
        for rec in tqdm(records, desc=f"OGA side={side}"):
            image = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            clean_pred = clean_predictions[rec.img_id]
            forbidden = list(rec.boxes) + list(clean_pred.boxes)
            # Deterministic per image and trigger size, matching the baseline's
            # reproducibility goal while avoiding a global-state RNG dependency.
            rng = np.random.default_rng(seed + rec.img_id * 1009 + int(side) * 9176)
            boxes = find_background_patch_boxes(
                rec.width, rec.height, forbidden, [int(side)], rng, max_trials=1000
            )
            if not boxes:
                continue
            pbox = boxes[0]
            poisoned = paste_patch_numpy_bgr(image, patch, pbox, blend_ratio)
            post = predict(poisoned)
            hit = False
            for pb, pl in zip(post.boxes, post.labels):
                if int(pl) == int(target_label) and rect_intersects(pbox, pb):
                    hit = True
                    break
            success += int(hit)
            attempted += 1
        asr = success / attempted if attempted else float("nan")
        results.append({"scenario": "oga", "trigger_size": int(side), "asr": asr,
                        "success": success, "attempted": attempted})
    return results


def write_results(rows: Sequence[dict], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "trigger_size", "asr", "success", "attempted"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"saved: {output_csv}")
