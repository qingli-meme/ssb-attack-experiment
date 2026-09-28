from __future__ import annotations

import argparse

import numpy as np
from mmdet.apis import inference_detector, init_detector

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
    p = argparse.ArgumentParser("Evaluate SSB ODA/OGA on MMDetection models")
    p.add_argument("--scenario", choices=["oda", "oga"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--trigger_ckpt", required=True)
    p.add_argument("--coco_root", default="datasets/coco/clean")
    p.add_argument("--trigger_sizes", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 80, 100])
    p.add_argument("--target_class", default="hair drier")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score_threshold", type=float, default=0.3)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--bbox_to_trigger_ratio", type=float, default=5.0)
    p.add_argument("--blend_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_images", type=int, default=-1)
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def parse_mmdet_result(result, score_threshold: float) -> Prediction:
    # MMDetection 3.x: DetDataSample with pred_instances.
    if hasattr(result, "pred_instances"):
        inst = result.pred_instances
        boxes = inst.bboxes.detach().cpu().numpy().astype(np.float32)
        labels = inst.labels.detach().cpu().numpy().astype(np.int64)
        scores = inst.scores.detach().cpu().numpy().astype(np.float32)
        keep = scores >= score_threshold
        return Prediction(boxes[keep], labels[keep], scores[keep])

    # MMDetection 2.x compatibility: list[class] -> ndarray [N,5].
    if isinstance(result, tuple):
        result = result[0]
    boxes_all, labels_all, scores_all = [], [], []
    for cls_idx, arr in enumerate(result):
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        keep = arr[:, 4] >= score_threshold
        arr = arr[keep]
        if arr.size == 0:
            continue
        boxes_all.append(arr[:, :4])
        scores_all.append(arr[:, 4])
        labels_all.append(np.full((arr.shape[0],), cls_idx, dtype=np.int64))
    if not boxes_all:
        return Prediction(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    return Prediction(
        np.concatenate(boxes_all, axis=0).astype(np.float32),
        np.concatenate(labels_all, axis=0),
        np.concatenate(scores_all, axis=0).astype(np.float32),
    )


def main():
    args = parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    trigger = load_trigger(args.trigger_ckpt, device="cpu")
    trigger.eval()
    records = load_records(args.coco_root, args.max_images)

    def predict(image_bgr: np.ndarray) -> Prediction:
        result = inference_detector(model, image_bgr)
        return parse_mmdet_result(result, args.score_threshold)

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
