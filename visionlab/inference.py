"""Lightweight YOLO inference with OpenVINO, without PyTorch or Ultralytics.

The models were trained by Ultralytics and exported once to OpenVINO IR
(see ``scripts/export_models.py``). At runtime only OpenVINO, NumPy and
OpenCV are needed, which keeps the app's memory around a quarter of what
loading PyTorch would cost: important on free hosting with tight RAM limits.

Pre- and post-processing mirror Ultralytics' own predictor (letterbox,
confidence filter, class-aware NMS, coordinate rescaling), and
``tests/test_inference.py`` checks the results on reference images.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visionlab.cpu import inference_threads
from visionlab.draw import draw_detections

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip

STRIDE = 32
PAD_VALUE = 114
NUM_KEYPOINTS = 17


@dataclass
class Detections:
    """Detections in original-image pixel coordinates."""

    boxes: np.ndarray  # (N, 4) float32, x1 y1 x2 y2
    scores: np.ndarray  # (N,) float32
    classes: np.ndarray  # (N,) int
    keypoints: np.ndarray | None = None  # (N, 17, 3) float32: x, y, confidence

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class Prediction:
    annotated_bgr: np.ndarray
    labels: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    detections: Detections | None = None

    def draw(self, image_bgr: np.ndarray) -> np.ndarray:
        """Draw these detections onto another (e.g. newer) frame of the same size, in place."""
        if self.detections is None:
            return image_bgr
        return draw_detections(image_bgr, self.detections, COCO_NAMES)


# --------------------------------------------------------------- pre/post --
def letterbox(
    image: np.ndarray, imgsz: int, stride: int = STRIDE
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize keeping aspect ratio, then pad to the smallest multiple of ``stride``.

    Matches Ultralytics' ``LetterBox(auto=True)`` used for dynamic-shape models.
    Returns the padded image and (new_w, new_h, pad_left, pad_top).
    """
    h, w = image.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (imgsz - new_w) % stride / 2, (imgsz - new_h) % stride / 2
    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE)
    )
    return padded, (new_w, new_h, left, top)


def to_tensor(image_bgr: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> 1x3xHxW RGB float32 in [0, 1]."""
    rgb = image_bgr[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices sorted by score."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    output: np.ndarray,
    num_classes: int,
    conf: float,
    iou: float = 0.5,
    max_det: int = 300,
    max_wh: int = 7680,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLO head output of shape (1, 4 + classes + extra, anchors).

    Returns boxes (x1y1x2y2, model input pixels), scores, classes and the extra
    per-anchor values (keypoints for pose models), after class-aware NMS.
    """
    pred = output[0].T  # (anchors, 4 + nc + extra)
    class_scores = pred[:, 4 : 4 + num_classes]
    classes = class_scores.argmax(1)
    scores = class_scores[np.arange(len(pred)), classes]
    keep = scores > conf
    pred, scores, classes = pred[keep], scores[keep], classes[keep]

    xy, wh = pred[:, 0:2], pred[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    # Offset boxes by class so NMS only suppresses overlaps within the same class.
    kept = nms(boxes + classes[:, None] * max_wh, scores, iou)[:max_det]
    return boxes[kept], scores[kept], classes[kept], pred[kept, 4 + num_classes :]


def scale_to_original(points: np.ndarray, letterbox_info, original_shape) -> np.ndarray:
    """Map x/y columns from letterboxed-input pixels back to the original image."""
    new_w, new_h, left, top = letterbox_info
    h0, w0 = original_shape[:2]
    points = points.copy()
    points[..., 0] = ((points[..., 0] - left) / (new_w / w0)).clip(0, w0)
    points[..., 1] = ((points[..., 1] - top) / (new_h / h0)).clip(0, h0)
    return points


# ----------------------------------------------------------------- engine --
class Detector:
    """A YOLO model running on OpenVINO's CPU plugin.

    One instance is cached per model and shared by every browser session, so a
    lock serialises inference. Threads are sized to the container's real CPU
    budget (see ``visionlab.cpu``).
    """

    device = "cpu"
    backend = "OpenVINO"

    def __init__(self, model_dir: str | Path, task: str, threads: int | None = None):
        import openvino as ov

        xml = next(Path(model_dir).glob("*.xml"))
        self.task = task
        self.num_classes = 1 if task == "pose" else len(COCO_NAMES)
        self.threads = threads or inference_threads()
        core = ov.Core()
        self._compiled = core.compile_model(
            core.read_model(str(xml)),
            "CPU",
            {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": self.threads},
        )
        self._request = self._compiled.create_infer_request()
        self._lock = threading.Lock()

    def warmup(self, imgsz: int = 320) -> None:
        """Run one dummy frame so the first real frame isn't slow."""
        self.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), conf=0.5, imgsz=imgsz)

    def detect(self, image_bgr: np.ndarray, conf: float, imgsz: int, iou: float = 0.5) -> Detections:
        padded, info = letterbox(image_bgr, imgsz)
        with self._lock:
            output = self._request.infer({0: to_tensor(padded)})[0]
        boxes, scores, classes, extra = postprocess(output, self.num_classes, conf, iou)

        boxes = scale_to_original(boxes.reshape(-1, 2, 2), info, image_bgr.shape).reshape(-1, 4)
        keypoints = None
        if self.task == "pose":
            keypoints = scale_to_original(extra.reshape(-1, NUM_KEYPOINTS, 3), info, image_bgr.shape)
        return Detections(boxes.astype(np.float32), scores.astype(np.float32), classes, keypoints)

    def predict(self, image_bgr: np.ndarray, conf: float, imgsz: int) -> Prediction:
        """Detect, then return the annotated frame, labels and timing."""
        start = time.perf_counter()
        detections = self.detect(image_bgr, conf, imgsz)
        latency_ms = (time.perf_counter() - start) * 1000
        labels = [COCO_NAMES[int(c)] for c in detections.classes]
        return Prediction(
            annotated_bgr=draw_detections(image_bgr.copy(), detections, COCO_NAMES),
            labels=labels,
            confidences=[float(s) for s in detections.scores],
            latency_ms=latency_ms,
            detections=detections,
        )
