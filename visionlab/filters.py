"""Classic (non-neural) OpenCV image filters."""

from __future__ import annotations

import cv2
import numpy as np


def pencil_sketch(image_bgr: np.ndarray, detail: int = 11, invert: bool = False) -> np.ndarray:
    """Turn a BGR frame into a sketch-like line drawing.

    Grayscale conversion followed by mean adaptive thresholding, which picks a
    separate threshold for each neighbourhood, so edges survive uneven
    lighting. A light median blur first suppresses webcam sensor noise.

    Args:
        image_bgr: HxWx3 uint8 image in OpenCV BGR order.
        detail: Neighbourhood size in pixels for the adaptive threshold.
            Smaller values give finer, busier lines. Must be odd and >= 3;
            even values are rounded up.
        invert: If True, draw white lines on black (the "blueprint" look)
            instead of dark lines on white.

    Returns:
        HxWx3 uint8 BGR image, so it can be streamed or encoded like any frame.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR image, got shape {image_bgr.shape}")

    block_size = max(3, int(detail))
    if block_size % 2 == 0:
        block_size += 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    lines = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block_size, 2)
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
