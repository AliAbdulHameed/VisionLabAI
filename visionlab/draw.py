"""Draw boxes, labels and pose skeletons in Ultralytics' visual style."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from visionlab.inference import Detections

# Ultralytics' default class palette (RGB hex), indexed by class id.
_PALETTE_HEX = (
    "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
    "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B",
)  # fmt: skip
PALETTE_BGR = [tuple(int(h[i : i + 2], 16) for i in (4, 2, 0)) for h in _PALETTE_HEX]

# COCO keypoint skeleton (0-indexed pairs) and Ultralytics' pose colors.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)  # fmt: skip
_POSE_RGB = np.array(
    [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255], [153, 204, 255],
     [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
     [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51], [0, 255, 0], [0, 0, 255],
     [255, 0, 0], [255, 255, 255]]
)  # fmt: skip
_LIMB_IDX = (9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16)
_KPT_IDX = (16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9)
LIMB_BGR = [tuple(int(v) for v in _POSE_RGB[i][::-1]) for i in _LIMB_IDX]
KPT_BGR = [tuple(int(v) for v in _POSE_RGB[i][::-1]) for i in _KPT_IDX]
KPT_CONF = 0.5  # hide keypoints the model isn't sure are visible


def line_width(image: np.ndarray) -> int:
    return max(round(sum(image.shape[:2]) / 2 * 0.003), 2)


def text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Dark text on light label backgrounds, white text on dark ones."""
    b, g, r = bgr
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 160 else (255, 255, 255)


def draw_box(image: np.ndarray, box: np.ndarray, label: str, color: tuple[int, int, int], lw: int) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)
    font_scale, thickness = lw / 3, max(lw - 1, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    th += 3
    outside = y1 >= th  # label above the box if there's room, otherwise inside it
    y_text = y1 - 2 if outside else y1 + th - 1
    cv2.rectangle(image, (x1, y1 - th if outside else y1), (x1 + tw, y1 if outside else y1 + th), color, -1)
    cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color(color),
                thickness, cv2.LINE_AA)


def draw_skeleton(image: np.ndarray, keypoints: np.ndarray, lw: int) -> None:
    radius = max(lw, 3)
    visible = keypoints[:, 2] >= KPT_CONF
    for (a, b), color in zip(SKELETON, LIMB_BGR, strict=True):
        if visible[a] and visible[b]:
            pa = (int(keypoints[a, 0]), int(keypoints[a, 1]))
            pb = (int(keypoints[b, 0]), int(keypoints[b, 1]))
            cv2.line(image, pa, pb, color, max(lw // 1, 2), cv2.LINE_AA)
    for (x, y, _), color, show in zip(keypoints, KPT_BGR, visible, strict=True):
        if show:
            cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def draw_detections(image: np.ndarray, detections: Detections, names: Sequence[str]) -> np.ndarray:
    """Draw all detections onto ``image`` in place and return it."""
    lw = line_width(image)
    # Lowest confidence first, so the most confident boxes end up on top.
    for i in np.argsort(detections.scores):
        cls = int(detections.classes[i])
        label = f"{names[cls]} {detections.scores[i]:.2f}"
        draw_box(image, detections.boxes[i], label, PALETTE_BGR[cls % 20], lw)
        if detections.keypoints is not None:
            draw_skeleton(image, detections.keypoints[i], lw)
    return image
