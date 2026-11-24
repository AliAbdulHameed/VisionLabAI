"""YOLO inference wrapper shared by the live stream and the image tab."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


def pick_device() -> int | str:
    """Use the first CUDA GPU when available, otherwise the CPU."""
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


@dataclass
class Prediction:
    annotated_bgr: np.ndarray
    labels: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    latency_ms: float = 0.0


class Detector:
    """Thread-safe wrapper around an Ultralytics YOLO model.

    One instance is cached per weights file and shared by every browser
    session, so a lock serialises calls into the model: Ultralytics
    predictors are not safe to call from several threads at once.
    """

    def __init__(self, weights: str, device: int | str | None = None):
        from ultralytics import YOLO

        self.weights = weights
        self.device = pick_device() if device is None else device
        self.model = YOLO(weights)
        self._lock = threading.Lock()

    def warmup(self, imgsz: int = 320) -> None:
        """Run one dummy frame so the first real frame isn't slow."""
        self.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), conf=0.5, imgsz=imgsz)

    def predict(self, image_bgr: np.ndarray, conf: float, imgsz: int) -> Prediction:
        """Run the model on a BGR frame and return the annotated frame and labels."""
        start = time.perf_counter()
        with self._lock:
            result = self.model.predict(
                source=image_bgr,
                conf=conf,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )[0]
        latency_ms = (time.perf_counter() - start) * 1000

        labels: list[str] = []
        confidences: list[float] = []
        if result.boxes is not None and len(result.boxes):
            labels = [result.names[int(c)] for c in result.boxes.cls.tolist()]
            confidences = [float(c) for c in result.boxes.conf.tolist()]

        return Prediction(
            annotated_bgr=result.plot(),
            labels=labels,
            confidences=confidences,
            latency_ms=latency_ms,
        )
