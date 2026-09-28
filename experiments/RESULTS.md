# SSB COCO subset experiment (2026-09-28)

## Scope and setup

The supplied archive contains SSB trigger optimization, poisoning, and evaluation code, but no detector training recipe or experiment README. This run uses the archive's defaults where available and documents the choices needed to complete a controlled experiment.

- Source: COCO 2017 images and annotations already available at `/home/icdm/lyx/LMPT/data/coco`.
- Fixed subset, seed 3407: 2,048 training images and 256 validation images. All 189 training and all 9 validation images containing `hair drier` are included; other images are sampled uniformly. See `datasets/coco/pilot/subset_metadata.json`.
- Hardware: two NVIDIA GeForce RTX 4090 D GPUs. Python 3.12, PyTorch 2.5.1+cu124, TorchVision 0.20.1+cu124, Ultralytics 8.4.164.
- Trigger optimization: Faster R-CNN ResNet50 FPN V2 COCO surrogate, 3,000 steps per scenario, 3 scales per step, batch size 1, reference size 30, scale range 0.5–2.0. Other arguments use the supplied script defaults. Checkpoints: `runs/pilot/trigger_oda_3000.pt` and `runs/pilot/trigger_oga_3000.pt`.
- Poisoning: ODA rate 0.10, 5 background triggers; OGA rate 1.0 among target-class images. Trigger sides 15–60, blend ratio 1.0. ODA modified 201/2,048 images with 990 inserts; OGA modified 175/2,048 images with 182 inserts. Labels were left unchanged.
- Detector: YOLOv8n initialized from `yolov8n.pt`, trained separately on clean, ODA, and OGA images for 20 epochs, batch 16, image size 640, seed 3407. The same clean validation subset was used for each. Ultralytics defaults apply for the remaining training options. Training logs and checkpoints are in `runs/pilot/train20`.
- Evaluation: the supplied `eval_yolov8.py` metric, 256 validation images, image size 640, score threshold 0.3, IoU threshold 0.5, trigger sizes 10/20/30/40/50/60/80/100. Each poisoned model is compared with the clean model at the same epoch using the same optimized trigger.

The supplied optimizer initially crashed when an ODA image had no valid boxes. Its `load_sample` now reshapes empty boxes to `(0, 4)`. The subset preparation script uses hard links so Ultralytics finds labels under the subset path.

## Results using epoch 20 (`last.pt`)

All ASR entries are successes / eligible attempts. ODA measures disappearance of originally detected objects. OGA measures a `hair drier` prediction intersecting a background trigger.

| Trigger side | ODA clean model | ODA poisoned model | OGA clean model | OGA poisoned model |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 21/578 (3.6%) | 20/573 (3.5%) | 0/242 (0%) | 0/240 (0%) |
| 20 | 15/344 (4.4%) | 17/339 (5.0%) | 0/237 (0%) | 35/235 (14.9%) |
| 30 | 9/204 (4.4%) | 8/204 (3.9%) | 0/234 (0%) | 66/231 (28.6%) |
| 40 | 3/131 (2.3%) | 8/137 (5.8%) | 0/231 (0%) | 84/228 (36.8%) |
| 50 | 2/89 (2.2%) | 7/96 (7.3%) | 0/224 (0%) | 94/222 (42.3%) |
| 60 | 2/58 (3.4%) | 4/63 (6.3%) | 1/219 (0.5%) | 71/219 (32.4%) |
| 80 | 2/18 (11.1%) | 6/23 (26.1%) | 0/209 (0%) | 11/207 (5.3%) |
| 100 | 0/0 (undefined) | 0/0 (undefined) | 0/197 (0%) | 12/197 (6.1%) |

The OGA effect is clear on this subset at 20–60 pixels. ODA changes are small at 10–60 pixels; the 80-pixel estimate has only 23 eligible objects, and 100 pixels has none. The ODA result does not establish a reliable disappearance effect.

Validation mAP50 at epoch 20: clean 0.4682, ODA 0.4792, OGA 0.4768. Corresponding mAP50-95: 0.3407, 0.3406, 0.3436.

Ultralytics selected epoch 1 as `best.pt` for all three models by mAP50-95. At those early checkpoints OGA ASR is 0% at every size for both models. The strong OGA result therefore depends on continued training despite a declining overall validation mAP. The best-checkpoint CSVs are also saved in `runs/pilot`.

## Reproduction

Run from the `ssb_attack` directory. The local `deps` directory contains Ultralytics and Polars; use `PYTHONPATH=deps` for YOLO commands. The `datasets/coco/clean` symlinks point to the existing COCO copy.

```bash
python experiments/prepare_subset.py --source datasets/coco/clean --output datasets/coco/pilot --train 2048 --val 256 --seed 3407
python -m source.ssb.optimize_trigger --scenario oda --coco_root datasets/coco/pilot --output runs/pilot/trigger_oda_3000.pt --device cuda:0 --steps 3000
python -m source.ssb.optimize_trigger --scenario oga --coco_root datasets/coco/pilot --output runs/pilot/trigger_oga_3000.pt --device cuda:1 --steps 3000
python -m source.ssb.poison_coco --scenario oda --trigger_ckpt runs/pilot/trigger_oda_3000.pt --coco_root datasets/coco/pilot --output_root runs/pilot/poison_oda_3000 --poison_rate 0.10
python -m source.ssb.poison_coco --scenario oga --trigger_ckpt runs/pilot/trigger_oga_3000.pt --coco_root datasets/coco/pilot --output_root runs/pilot/poison_oga_3000 --poison_rate 1.0
```

Train three YOLOv8n models for 20 epochs using `datasets/coco/pilot/coco_clean.yaml` and the two generated `coco_ssb.yaml` files. The exact arguments are recorded in `runs/pilot/train20_*.log`; the dataset and hyperparameters are listed above. Evaluate `last.pt` with `python -m source.ssb.eval_yolov8`, using `--imgsz 640 --max_images 256`. The four epoch-20 CSVs are `runs/pilot/asr_oda_last.csv`, `asr_oda_clean_last.csv`, `asr_oga_last.csv`, and `asr_oga_clean_last.csv`.

Copies of the ASR CSVs and trigger checkpoints are committed under `experiments/results` and `experiments/triggers`. Detector weights and COCO images remain local because they are generated or external data.

This is a fixed subset experiment. It does not estimate full-COCO performance or performance for other model families. The archive provides MMDetection evaluation code, but no MMDetection checkpoint or training configuration was supplied, so this run covers YOLOv8n only.
