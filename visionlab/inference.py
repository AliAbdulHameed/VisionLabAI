"""YOLO inference wrapper shared by the live stream and the image tab."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def cgroup_cpu_quota(root: str = "/sys/fs/cgroup", proc_cgroup: str = "/proc/self/cgroup") -> float | None:
    """CPUs this container may use according to its cgroup quota, or None if unlimited.

    Cloud hosts usually cap a container's CPU time with a quota while still
    showing it every core on the machine. Libraries then start one thread per
    visible core, burn through the quota in a burst and get frozen for the rest
    of the scheduling period, which shows up as stutter.
    """
    candidates = [Path(root, "cpu.max")]  # cgroup v2: "<quota> <period>" or "max <period>"
    try:
        for line in Path(proc_cgroup).read_text().splitlines():  # cgroup v1: "N:cpu,cpuacct:/path"
            _, controllers, path = line.split(":", 2)
            if "cpu" in controllers.split(","):
                for mount in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
                    candidates += [Path(root, mount, path.lstrip("/")), Path(root, mount)]
    except (OSError, ValueError):
        pass

    for candidate in candidates:
        try:
            if candidate.name == "cpu.max":
                quota, period = candidate.read_text().split()
                if quota != "max":
                    return int(quota) / int(period)
                return None
            quota = int((candidate / "cpu.cfs_quota_us").read_text())
            period = int((candidate / "cpu.cfs_period_us").read_text())
            return quota / period if quota > 0 else None
        except (OSError, ValueError):
            continue
    return None


def cpu_budget() -> int:
    """Whole CPUs actually available: visible cores, capped by any cgroup quota."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, math.floor(quota))
    return max(1, cpus)


def inference_threads(budget: int | None = None) -> int:
    """Threads for the model: the CPU budget minus one core left for video encoding."""
    budget = cpu_budget() if budget is None else budget
    return max(1, budget - 1)


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
    result: Any = None  # the raw Ultralytics Results, kept so it can be redrawn

    def draw(self, image_bgr: np.ndarray) -> np.ndarray:
        """Draw these detections onto another (e.g. newer) frame of the same size."""
        if self.result is None:
            return image_bgr
        return self.result.plot(img=image_bgr)


def export_openvino(model) -> str | None:
    """Convert a PyTorch YOLO model to OpenVINO, reusing an earlier export.

    OpenVINO runs these nano models 3-4x faster than PyTorch on Intel/AMD
    CPUs. The export is dynamic so the inference resolution can change at
    runtime. Returns the model directory, or None if OpenVINO is unavailable.
    """
    try:
        import openvino  # noqa: F401
    except ImportError:
        return None

    weights = str(model.ckpt_path or model.model_name)
    out_dir = weights.rsplit(".", 1)[0] + "_openvino_model"
    if os.path.isfile(os.path.join(out_dir, "metadata.yaml")):
        return out_dir
    try:
        return str(model.export(format="openvino", dynamic=True, verbose=False))
    except Exception:  # fall back to PyTorch rather than take the app down
        LOGGER.exception("OpenVINO export failed for %s; using PyTorch", weights)
        return None


def limit_openvino_threads(yolo, ov_dir: str, threads: int) -> bool:
    """Recompile an Ultralytics OpenVINO model with a fixed CPU thread count.

    Ultralytics compiles OpenVINO models with its own ``Core`` and no thread
    limit, so this reaches into its backend after the first prediction. If the
    internals ever change, it logs a warning and leaves the model as it was.
    """
    try:
        backend = yolo.predictor.model.backend
        compile_fn = backend.compile_model  # partial(core.compile_model, device_name=..., config=...)
        config = {**compile_fn.keywords.get("config", {}), "INFERENCE_NUM_THREADS": threads}
        keywords = {**compile_fn.keywords, "config": config}
        compile_limited = partial(compile_fn.func, *compile_fn.args, **keywords)
        core = compile_fn.func.__self__
        xml = next(Path(ov_dir).glob("*.xml"))
        compiled = compile_limited(core.read_model(str(xml)))
        backend.compile_model, backend.ov_compiled_model = compile_limited, compiled
        return int(compiled.get_property("INFERENCE_NUM_THREADS")) == threads
    except Exception:
        LOGGER.warning("Could not limit OpenVINO threads; using its defaults", exc_info=True)
        return False


class Detector:
    """Thread-safe wrapper around an Ultralytics YOLO model.

    One instance is cached per weights file and shared by every browser
    session, so a lock serialises calls into the model: Ultralytics
    predictors are not safe to call from several threads at once.

    On CPU the model runs through OpenVINO when available (set
    ``VISIONLAB_BACKEND=torch`` to force PyTorch); on a GPU it uses PyTorch.
    """

    def __init__(self, weights: str, device: int | str | None = None, backend: str | None = None):
        from ultralytics import YOLO

        self.weights = weights
        self.device = pick_device() if device is None else device
        backend = (backend or os.getenv("VISIONLAB_BACKEND", "auto")).lower()

        model = YOLO(weights)
        self.backend = "PyTorch"
        self.threads = inference_threads()
        ov_dir = None
        if self.device == "cpu" and backend in ("auto", "openvino"):
            ov_dir = export_openvino(model)
            if ov_dir is not None:
                model = YOLO(ov_dir, task=model.task)
                self.backend = "OpenVINO"
        self.model = model
        self._lock = threading.Lock()

        if self.device == "cpu":
            if ov_dir is None:
                import torch

                torch.set_num_threads(self.threads)
            else:
                self.warmup()  # creates the Ultralytics predictor, which compiles the model
                limit_openvino_threads(self.model, ov_dir, self.threads)

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
            result=result,
        )
