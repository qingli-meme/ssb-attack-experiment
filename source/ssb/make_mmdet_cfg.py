from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config


def parse_args():
    p = argparse.ArgumentParser("Create a minimal MMDetection config that points to SSB poisoned images")
    p.add_argument("--base_config", required=True)
    p.add_argument("--poison_root", required=True,
                   help="e.g. /abs/path/Attacking-by-Aligning/datasets/coco/ssb_oda_p010")
    p.add_argument("--clean_coco_root", required=True,
                   help="e.g. /abs/path/Attacking-by-Aligning/datasets/coco/clean")
    p.add_argument("--output", required=True)
    p.add_argument("--lr", type=float, default=None,
                   help="optional optimizer LR override; Align paper used 1e-4")
    return p.parse_args()


def _unwrap_dataset(dataset_cfg):
    """Return the innermost dataset config for common MMDet wrappers."""
    cur = dataset_cfg
    # RepeatDataset/ClassBalancedDataset-style wrapper.
    while isinstance(cur, dict) and "dataset" in cur and isinstance(cur["dataset"], dict):
        cur = cur["dataset"]
    return cur


def set_dataset_paths(loader_cfg, ann_file: str, image_dir: str):
    ds = _unwrap_dataset(loader_cfg["dataset"])
    ds["data_root"] = ""
    ds["ann_file"] = ann_file
    prefix = ds.get("data_prefix", {})
    if not isinstance(prefix, dict):
        prefix = {}
    prefix["img"] = image_dir.rstrip("/") + "/"
    ds["data_prefix"] = prefix


def main():
    args = parse_args()
    base = Path(args.base_config).resolve()
    poison = Path(args.poison_root).resolve()
    clean = Path(args.clean_coco_root).resolve()

    cfg = Config.fromfile(str(base))
    train_ann = str(clean / "annotations" / "instances_train2017.json")
    val_ann = str(clean / "annotations" / "instances_val2017.json")
    poison_train = str(poison / "images" / "train2017")
    clean_val = str(clean / "images" / "val2017")

    set_dataset_paths(cfg.train_dataloader, train_ann, poison_train)
    set_dataset_paths(cfg.val_dataloader, val_ann, clean_val)
    set_dataset_paths(cfg.test_dataloader, val_ann, clean_val)

    # Evaluators need the clean COCO annotations as well.
    for key in ("val_evaluator", "test_evaluator"):
        if key in cfg:
            ev = cfg[key]
            if isinstance(ev, dict) and "ann_file" in ev:
                ev["ann_file"] = val_ann
            elif isinstance(ev, (list, tuple)):
                for item in ev:
                    if isinstance(item, dict) and "ann_file" in item:
                        item["ann_file"] = val_ann

    if args.lr is not None:
        try:
            cfg.optim_wrapper.optimizer.lr = float(args.lr)
        except Exception as exc:
            raise RuntimeError("Could not override cfg.optim_wrapper.optimizer.lr") from exc

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
