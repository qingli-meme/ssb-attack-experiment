from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch import Tensor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from source.ssb.common import (
    centered_patch_box,
    find_background_patch_boxes,
    paste_patch_tensor_rgb,
    sample_log_uniform,
    seed_everything,
    xywh_to_xyxy,
)
from source.ssb.trigger import ScaleSteerableTrigger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Learn Scale-Steerable Basis trigger coefficients")
    p.add_argument("--scenario", choices=["oda", "oga"], required=True)
    p.add_argument("--coco_root", default="datasets/coco/clean")
    p.add_argument("--split", default="train2017")
    p.add_argument("--target_class", default="hair drier")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--scales_per_step", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--beta", type=float, default=8.0,
                   help="smooth worst-scale temperature; larger -> closer to max")

    p.add_argument("--reference_size", type=int, default=30)
    p.add_argument("--min_scale", type=float, default=0.5,
                   help="minimum render side / reference_size during trigger learning")
    p.add_argument("--max_scale", type=float, default=2.0)
    p.add_argument("--num_trigger", type=int, default=5,
                   help="ODA: number of background triggers per image")
    p.add_argument("--blend_ratio", type=float, default=1.0)

    p.add_argument("--radial_frequencies", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--angular_orders", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--contrast", type=float, default=1.75)
    p.add_argument("--log_every", type=int, default=25)
    return p.parse_args()


def load_sample(coco: COCO, coco_root: Path, split: str, img_id: int, device: torch.device):
    info = coco.loadImgs([img_id])[0]
    path = coco_root / "images" / split / info["file_name"]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0

    boxes, labels = [], []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None)):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        boxes.append([x, y, x + w, y + h])
        # TorchVision's COCO detector uses the original COCO category IDs (1..90 with gaps).
        labels.append(int(ann["category_id"]))

    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        "labels": torch.as_tensor(labels, dtype=torch.int64, device=device),
    }
    return image, target, info


def poison_one(
    image: Tensor,
    target: Dict[str, Tensor],
    trigger: ScaleSteerableTrigger,
    scenario: str,
    side: int,
    target_category_id: int | None,
    num_trigger: int,
    blend_ratio: float,
    rng: np.random.Generator,
) -> tuple[Tensor, int]:
    _, height, width = image.shape
    boxes_np = target["boxes"].detach().cpu().numpy()
    labels_np = target["labels"].detach().cpu().numpy()
    patch = trigger.render(side)
    out = image
    inserted = 0

    if scenario == "oda":
        sides = [side] * num_trigger
        positions = find_background_patch_boxes(width, height, boxes_np, sides, rng)
        for box in positions:
            out = paste_patch_tensor_rgb(out, patch, box, blend_ratio)
            inserted += 1
        return out, inserted

    # OGA: insert into target-class GT boxes, keep annotations unchanged.
    assert target_category_id is not None
    for box, label in zip(boxes_np, labels_np):
        if int(label) != int(target_category_id):
            continue
        x1, y1, x2, y2 = box
        if side > (x2 - x1) or side > (y2 - y1):
            continue
        pbox = centered_patch_box(box, side)
        px1, py1, px2, py2 = pbox
        if px1 < 0 or py1 < 0 or px2 > width or py2 > height:
            continue
        out = paste_patch_tensor_rgb(out, patch, pbox, blend_ratio)
        inserted += 1
    return out, inserted


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    coco_root = Path(args.coco_root)
    ann_path = coco_root / "annotations" / f"instances_{args.split}.json"
    coco = COCO(str(ann_path))

    target_category_id = None
    if args.scenario == "oga":
        cat_ids = coco.getCatIds(catNms=[args.target_class])
        if len(cat_ids) != 1:
            raise RuntimeError(f"Cannot resolve target class {args.target_class!r}")
        target_category_id = int(cat_ids[0])
        image_ids = sorted(set(coco.getImgIds(catIds=[target_category_id])))
    else:
        image_ids = sorted(coco.getImgIds())

    if not image_ids:
        raise RuntimeError("No eligible images found")

    trigger = ScaleSteerableTrigger(
        reference_size=args.reference_size,
        radial_frequencies=args.radial_frequencies,
        angular_orders=args.angular_orders,
        contrast=args.contrast,
    ).to(device)
    optimizer = torch.optim.Adam([trigger.coeff], lr=args.lr)

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    surrogate = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    surrogate.train()  # detection models return training losses only in train mode
    for param in surrogate.parameters():
        param.requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    running = []

    for step in range(1, args.steps + 1):
        replace = len(image_ids) < args.batch_size
        batch_ids = rng.choice(image_ids, size=args.batch_size, replace=replace).tolist()
        clean_batch = [load_sample(coco, coco_root, args.split, int(i), device) for i in batch_ids]

        scale_losses: List[Tensor] = []
        scale_values: List[float] = []
        for _ in range(args.scales_per_step):
            scale_ratio = sample_log_uniform(rng, args.min_scale, args.max_scale)
            side = max(4, int(round(args.reference_size * scale_ratio)))
            poisoned_images, poisoned_targets = [], []
            inserted_total = 0

            for image, target, _ in clean_batch:
                poisoned, inserted = poison_one(
                    image=image,
                    target=target,
                    trigger=trigger,
                    scenario=args.scenario,
                    side=side,
                    target_category_id=target_category_id,
                    num_trigger=args.num_trigger,
                    blend_ratio=args.blend_ratio,
                    rng=rng,
                )
                if inserted > 0:
                    poisoned_images.append(poisoned)
                    poisoned_targets.append(target)
                    inserted_total += inserted

            # OGA may have target objects too small for an extreme sampled side.
            if inserted_total == 0:
                continue

            loss_dict = surrogate(poisoned_images, poisoned_targets)
            det_loss = torch.stack([v for v in loss_dict.values()]).sum()
            scale_losses.append(det_loss)
            scale_values.append(scale_ratio)

        if not scale_losses:
            continue

        losses = torch.stack(scale_losses)
        # Normalized LogSumExp approximates max(loss_s) while staying smooth.
        beta = float(args.beta)
        scale_objective = torch.logsumexp(beta * losses, dim=0) / beta
        scale_objective = scale_objective - math.log(len(scale_losses)) / beta

        optimizer.zero_grad(set_to_none=True)
        scale_objective.backward()
        torch.nn.utils.clip_grad_norm_([trigger.coeff], max_norm=10.0)
        optimizer.step()

        running.append(float(scale_objective.detach().cpu()))
        if step % args.log_every == 0 or step == 1:
            avg = sum(running[-args.log_every:]) / min(len(running), args.log_every)
            print(
                f"step={step:05d} objective={float(scale_objective):.4f} "
                f"avg={avg:.4f} scales={[round(s, 3) for s in scale_values]}"
            )

    trigger.save(
        args.output,
        extra={
            "scenario": args.scenario,
            "target_class": args.target_class,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "optimizer": "Adam",
            "lr": args.lr,
            "steps": args.steps,
            "surrogate": "torchvision fasterrcnn_resnet50_fpn_v2 COCO weights",
            "seed": args.seed,
        },
    )
    print(f"saved trigger to {args.output}")


if __name__ == "__main__":
    main()
