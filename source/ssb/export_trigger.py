from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from source.ssb.trigger import load_trigger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trigger_ckpt", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 80, 100])
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    trigger = load_trigger(args.trigger_ckpt, device="cpu")
    trigger.eval()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for side in args.sizes:
        patch = trigger.render_numpy_bgr(side)
        cv2.imwrite(str(out / f"trigger_{side:03d}.png"), (patch * 255.0).round().astype("uint8"))


if __name__ == "__main__":
    main()
