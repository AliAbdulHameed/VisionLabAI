"""Heads-up display drawn on top of live video frames."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Sequence

import cv2
import numpy as np


class FPSMeter:
    """Frames-per-second estimate, smoothed with an exponential moving average."""

    def __init__(self, smoothing: float = 0.9, clock=time.perf_counter):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self._smoothing = smoothing
        self._clock = clock
        self._last: float | None = None
        self.fps: float = 0.0

    def tick(self) -> float:
        """Record a frame and return the current smoothed FPS."""
        now = self._clock()
        if self._last is not None:
            elapsed = now - self._last
            if elapsed > 0:
                instant = 1.0 / elapsed
                self.fps = instant if self.fps == 0 else (
                    self._smoothing * self.fps + (1 - self._smoothing) * instant
                )
        self._last = now
        return self.fps


def summarize_counts(labels: Iterable[str], limit: int = 4) -> str:
    """Summarize detected labels, most common first: ``"2 person · 1 tv"``."""
    counts = Counter(labels)
    if not counts:
        return "nothing detected"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{count} {label}" for label, count in ranked[:limit]]
    if len(ranked) > limit:
        parts.append(f"+{len(ranked) - limit} more")
    return " · ".join(parts)


def draw_hud(
    image_bgr: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (10, 10),
    scale: float = 0.55,
) -> np.ndarray:
    """Draw text lines on a translucent dark panel in the top-left corner.

    Draws in place and also returns the image for convenience.
    """
    if not lines:
        return image_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 8
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_height = max(h for _, h in sizes) + 8
    width = max(w for w, _ in sizes) + 2 * pad
    height = line_height * len(lines) + pad

    x0, y0 = origin
    x1 = min(x0 + width, image_bgr.shape[1])
    y1 = min(y0 + height, image_bgr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return image_bgr

    region = image_bgr[y0:y1, x0:x1]
    region[:] = (region * 0.35).astype(np.uint8)

    for i, line in enumerate(lines):
        baseline_y = y0 + pad + line_height * i + sizes[i][1]
        cv2.putText(image_bgr, line, (x0 + pad, baseline_y), font, scale,
                    (162, 201, 64), thickness, cv2.LINE_AA)
    return image_bgr
