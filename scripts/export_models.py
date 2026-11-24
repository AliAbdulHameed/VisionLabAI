"""Regenerate the OpenVINO models in ``models/`` from Ultralytics' pretrained weights.

Only needed when changing models; the app itself never imports PyTorch or
Ultralytics. Run from the repo root:

    pip install ultralytics openvino
    python scripts/export_models.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS = {"yolo11n": "yolo11n.pt", "yolo26n-pose": "yolo26n-pose.pt"}

for name, weights in MODELS.items():
    # Dynamic input shapes let the app change resolution at runtime; FP16 weights halve the file size.
    exported = Path(YOLO(weights).export(format="openvino", dynamic=True, half=True))
    target = Path("models") / name
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(exported, target)
    print(f"{weights} -> {target}")
