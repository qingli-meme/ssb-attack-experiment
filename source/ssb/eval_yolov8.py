from __future__ import annotations

import argparse

import numpy as np
from ultralytics import YOLO

from source.ssb.common import COCO80_NAMES
from source.ssb.eval_common import (
    Prediction,
    evaluate_oda,
    evaluate_oga,
    load_records,
    run_clean_predictions,
    write_results,
)
from source.ssb.trigger import load_trigger


def parse_args():
    p = argparse.ArgumentParser("Evaluate SSB ODA/OGA on Ultralytics YOLOv8")
    p.add_argument("--scenario", choices=["oda", "oga"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--trigger_ckpt", required=True)
    p.add_argument("--coco_root", default="datasets/coco/clean")
    p.add_argument("--trigger_sizes", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 80, 100])
    p.add_argument("--target_class", default="hair drier")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--score_threshold", type=float, default=0.3)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--bbox_to_trigger_ratio", type=float, default=5.0)
    p.add_argument("--blend_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_images", type=int, default=-1)
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    trigger = load_trigger(args.trigger_ckpt, device="cpu")
    trigger.eval()
    records = load_records(args.coco_root, args.max_images)

    def predict(image_bgr: np.ndarray) -> Prediction:
        result = model.predict(
            source=image_bgr,
            imgsz=args.imgsz,
            conf=args.score_threshold,
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return Prediction(
                boxes=np.zeros((0, 4), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int64),
                scores=np.zeros((0,), dtype=np.float32),
            )
        return Prediction(
            boxes=result.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
            labels=result.boxes.cls.detach().cpu().numpy().astype(np.int64),
            scores=result.boxes.conf.detach().cpu().numpy().astype(np.float32),
        )

    clean = run_clean_predictions(records, predict)
    if args.scenario == "oda":
        rows = evaluate_oda(
            records, clean, predict, trigger, args.trigger_sizes,
            args.blend_ratio, args.iou_threshold, args.bbox_to_trigger_ratio,
        )
    else:
        if args.target_class not in COCO80_NAMES:
            raise ValueError(f"Unknown COCO class: {args.target_class}")
        target_label = COCO80_NAMES.index(args.target_class)
        rows = evaluate_oga(
            records, clean, predict, trigger, args.trigger_sizes,
            target_label, args.blend_ratio, args.seed,
        )
    write_results(rows, args.output_csv)


if __name__ == "__main__":
    main()
