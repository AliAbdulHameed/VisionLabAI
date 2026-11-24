"""Heads-up display drawn on top of live video frames."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Sequence

import cv2
import numpy as np


class FPSMeter:
    """Frames-per-second estimate from an exponential moving average of frame gaps.

    Averaging the gaps and inverting (rather than averaging ``1 / gap``) keeps
    the estimate honest when frames arrive unevenly: a burst of closely spaced
    frames would otherwise inflate the reading well past the true frame rate.
    """

    def __init__(self, smoothing: float = 0.9, clock=time.perf_counter, reset_after_s: float = 2.0):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self._smoothing = smoothing
        self._clock = clock
        self._reset_after_s = reset_after_s
        self._last: float | None = None
        self._avg_gap: float | None = None

    @property
    def fps(self) -> float:
        return 1.0 / self._avg_gap if self._avg_gap else 0.0

    def tick(self) -> float:
        """Record a frame and return the current smoothed FPS."""
        now = self._clock()
        if self._last is not None:
            gap = now - self._last
            if gap > self._reset_after_s:
                # The stream was paused or restarted: start averaging afresh.
                self._avg_gap = None
            elif gap > 0:
                self._avg_gap = gap if self._avg_gap is None else (
                    self._smoothing * self._avg_gap + (1 - self._smoothing) * gap
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
    anchor: str = "top-left",
) -> np.ndarray:
    """Draw text lines on a translucent dark panel.

    ``origin`` is the margin from the anchored corner: "top-left" or
    "bottom-left" (which keeps the panel clear of box labels drawn at the
    top of detections). Draws in place and also returns the image.
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

    if anchor not in ("top-left", "bottom-left"):
        raise ValueError(f"Unsupported anchor: {anchor}")
    x0, y0 = origin
    if anchor == "bottom-left":
        y0 = max(0, image_bgr.shape[0] - height - origin[1])
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
