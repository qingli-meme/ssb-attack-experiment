from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from source.ssb.common import (
    centered_patch_box,
    find_background_patch_boxes,
    paste_patch_numpy_bgr,
    sample_side,
    seed_everything,
    xywh_to_xyxy,
    yolo_label_line,
)
from source.ssb.trigger import load_trigger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate clean-label COCO poison set with SSB trigger")
    p.add_argument("--scenario", choices=["oda", "oga"], required=True)
    p.add_argument("--trigger_ckpt", required=True)
    p.add_argument("--coco_root", default="datasets/coco/clean")
    p.add_argument("--output_root", required=True)
    p.add_argument("--target_class", default="hair drier")
    p.add_argument("--poison_rate", type=float, default=None,
                   help="ODA default 0.10; OGA default 1.0 among target-class images")
    p.add_argument("--num_trigger", type=int, default=5,
                   help="ODA: number of background triggers per poisoned image")
    p.add_argument("--min_trigger", type=int, default=15)
    p.add_argument("--max_trigger", type=int, default=60)
    p.add_argument("--blend_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_yolo_labels(coco: COCO, image_ids, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_ids = sorted(coco.getCatIds())
    cat_to_yolo = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    for img_id in tqdm(image_ids, desc=f"labels:{output_dir.name}"):
        info = coco.loadImgs([img_id])[0]
        stem = Path(info["file_name"]).stem
        with (output_dir / f"{stem}.txt").open("w") as f:
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None)):
                if ann.get("iscrowd", 0):
                    continue
                f.write(
                    yolo_label_line(
                        cat_to_yolo[int(ann["category_id"])],
                        ann["bbox"],
                        int(info["width"]),
                        int(info["height"]),
                    )
                )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.poison_rate is None:
        args.poison_rate = 0.10 if args.scenario == "oda" else 1.0
    if not (0.0 <= args.poison_rate <= 1.0):
        raise ValueError("poison_rate must be in [0,1]")

    clean_root = Path(args.coco_root)
    out_root = Path(args.output_root)
    train_out = out_root / "images" / "train2017"
    train_labels = out_root / "labels" / "train2017"
    clean_val_labels = clean_root / "labels" / "val2017"
    train_out.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    train_coco = COCO(str(clean_root / "annotations" / "instances_train2017.json"))
    val_coco = COCO(str(clean_root / "annotations" / "instances_val2017.json"))
    train_ids = sorted(train_coco.getImgIds())
    val_ids = sorted(val_coco.getImgIds())

    ensure_yolo_labels(train_coco, train_ids, train_labels)
    ensure_yolo_labels(val_coco, val_ids, clean_val_labels)

    target_cat_id = None
    target_image_ids = set()
    if args.scenario == "oga":
        ids = train_coco.getCatIds(catNms=[args.target_class])
        if len(ids) != 1:
            raise RuntimeError(f"Cannot resolve target class {args.target_class!r}")
        target_cat_id = int(ids[0])
        target_image_ids = set(train_coco.getImgIds(catIds=[target_cat_id]))

    trigger = load_trigger(args.trigger_ckpt, device=args.device)
    trigger.eval()

    selected_images = 0
    modified_images = 0
    inserted_triggers = 0

    train_txt = out_root / "train2017.txt"
    with train_txt.open("w") as list_file:
        for img_id in tqdm(train_ids, desc=f"poison:{args.scenario}"):
            info = train_coco.loadImgs([img_id])[0]
            src = clean_root / "images" / "train2017" / info["file_name"]
            dst = train_out / info["file_name"]
            list_file.write(str(dst.resolve()) + "\n")

            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(src)
            height, width = image.shape[:2]
            anns = [
                a for a in train_coco.loadAnns(train_coco.getAnnIds(imgIds=[img_id], iscrowd=None))
                if not a.get("iscrowd", 0)
            ]
            boxes = [xywh_to_xyxy(a["bbox"]) for a in anns]

            if args.scenario == "oda":
                eligible = rng.random() <= args.poison_rate
            else:
                eligible = img_id in target_image_ids and rng.random() <= args.poison_rate

            if not eligible:
                cv2.imwrite(str(dst), image)
                continue

            selected_images += 1
            poisoned = image
            inserted_here = 0

            if args.scenario == "oda":
                occupied = list(boxes)
                for _ in range(args.num_trigger):
                    side = sample_side(rng, args.min_trigger, args.max_trigger)
                    positions = find_background_patch_boxes(width, height, occupied, [side], rng)
                    if not positions:
                        continue
                    box = positions[0]
                    patch = trigger.render_numpy_bgr(side)
                    poisoned = paste_patch_numpy_bgr(poisoned, patch, box, args.blend_ratio)
                    occupied.append(np.asarray(box, dtype=np.float32))
                    inserted_here += 1
            else:
                assert target_cat_id is not None
                for ann, box in zip(anns, boxes):
                    if int(ann["category_id"]) != target_cat_id:
                        continue
                    x1, y1, x2, y2 = box
                    max_side = min(args.max_trigger, int(x2 - x1), int(y2 - y1))
                    if max_side < args.min_trigger:
                        continue
                    side = sample_side(rng, args.min_trigger, max_side)
                    pbox = centered_patch_box(box, side)
                    px1, py1, px2, py2 = pbox
                    if px1 < 0 or py1 < 0 or px2 > width or py2 > height:
                        continue
                    patch = trigger.render_numpy_bgr(side)
                    poisoned = paste_patch_numpy_bgr(poisoned, patch, pbox, args.blend_ratio)
                    inserted_here += 1

            if inserted_here > 0:
                modified_images += 1
                inserted_triggers += inserted_here
            cv2.imwrite(str(dst), poisoned)

    # Validation stays clean. Ultralytics can read an absolute image directory.
    val_dir = clean_root / "images" / "val2017"
    val_txt = out_root / "val2017.txt"
    with val_txt.open("w") as f:
        for img_id in val_ids:
            info = val_coco.loadImgs([img_id])[0]
            f.write(str((val_dir / info["file_name"]).resolve()) + "\n")

    # COCO annotations are untouched by construction. Keep a small metadata file
    # instead of copying 100+ MB annotation JSON files.
    metadata = {
        "scenario": args.scenario,
        "trigger_ckpt": str(Path(args.trigger_ckpt).resolve()),
        "coco_root": str(clean_root.resolve()),
        "target_class": args.target_class,
        "poison_rate_among_eligible": args.poison_rate,
        "selected_images": selected_images,
        "modified_images": modified_images,
        "inserted_triggers": inserted_triggers,
        "min_trigger": args.min_trigger,
        "max_trigger": args.max_trigger,
        "blend_ratio": args.blend_ratio,
        "seed": args.seed,
        "annotation_train": str((clean_root / "annotations" / "instances_train2017.json").resolve()),
        "annotation_val": str((clean_root / "annotations" / "instances_val2017.json").resolve()),
    }
    with (out_root / "ssb_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    yaml_path = out_root / "coco_ssb.yaml"
    with yaml_path.open("w") as f:
        f.write(f"train: {train_txt.resolve()}\n")
        f.write(f"val: {val_txt.resolve()}\n")
        f.write("nc: 80\n")
        f.write("names:\n")
        for i, name in enumerate([c["name"] for c in train_coco.loadCats(sorted(train_coco.getCatIds()))]):
            f.write(f"  {i}: {name!r}\n")

    print(json.dumps(metadata, indent=2))
    print(f"YOLO yaml: {yaml_path}")


if __name__ == "__main__":
    main()
