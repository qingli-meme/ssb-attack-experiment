"""Create a reproducible COCO detection subset using hard links to existing images."""

import argparse
import json
import random
from pathlib import Path

from pycocotools.coco import COCO


def make_split(source: Path, output: Path, split: str, count: int, seed: int) -> dict:
    coco = COCO(str(source / "annotations" / f"instances_{split}.json"))
    target_id = coco.getCatIds(catNms=["hair drier"])[0]
    target_images = set(coco.getImgIds(catIds=[target_id]))
    all_images = set(coco.getImgIds())
    if count < len(target_images):
        raise ValueError(f"{split}: count {count} is below {len(target_images)} target images")
    rng = random.Random(seed)
    rest = sorted(all_images - target_images)
    selected = target_images | set(rng.sample(rest, min(count - len(target_images), len(rest))))

    image_dir = output / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    for image in coco.loadImgs(sorted(selected)):
        src = (source / "images" / split / image["file_name"]).resolve(strict=True)
        dst = image_dir / image["file_name"]
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            dst.hardlink_to(src)

    annotations = [ann for ann in coco.dataset["annotations"] if ann["image_id"] in selected]
    data = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": coco.loadImgs(sorted(selected)),
        "annotations": annotations,
        "categories": coco.dataset["categories"],
    }
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    with (annotation_dir / f"instances_{split}.json").open("w") as f:
        json.dump(data, f)
    return {"images": len(selected), "target_images": len(target_images), "annotations": len(annotations)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=2048)
    parser.add_argument("--val", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "train2017": make_split(args.source, args.output, "train2017", args.train, args.seed),
        "val2017": make_split(args.source, args.output, "val2017", args.val, args.seed + 1),
    }
    (args.output / "subset_metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
