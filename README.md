# Scale-Steerable Basis attack experiment

This repository contains the SSB attack code supplied in `ssb_attack_code.zip`, a reproducible COCO subset preparation script, and the results of an ODA/OGA YOLOv8n experiment.

Start with [the experiment report](experiments/RESULTS.md). The report includes settings, metrics, controls, and limitations. The small trigger checkpoints and ASR CSV files are in `experiments/triggers` and `experiments/results`.

COCO images, installed dependencies, and detector weights are excluded from Git. The experiment used COCO 2017 and the package versions listed in the report. Run modules from this directory, for example `python -m source.ssb.optimize_trigger --help`.
