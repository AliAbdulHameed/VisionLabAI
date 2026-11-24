"""Thread-safe hand-off of live stream statistics to the UI.

The WebRTC frame callback runs on a worker thread and writes here; the
Streamlit UI reads periodic snapshots to render the live stats panel.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class LiveSnapshot:
    fps: float
    latency_ms: float
    counts: tuple[tuple[str, int], ...]  # (label, count), most common first
    frames: int
    age_s: float | None  # seconds since the last frame; None if no frames yet

    @property
    def total_objects(self) -> int:
        return sum(count for _, count in self.counts)

    def is_fresh(self, max_age_s: float = 2.0) -> bool:
        return self.age_s is not None and self.age_s <= max_age_s


class LiveStats:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._fps = 0.0
        self._latency_ms = 0.0
        self._counts: tuple[tuple[str, int], ...] = ()
        self._frames = 0
        self._updated: float | None = None

    def update(self, fps: float, latency_ms: float, labels: list[str] | None = None) -> None:
        counts = Counter(labels or ())
        ranked = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        with self._lock:
            self._fps = fps
            self._latency_ms = latency_ms
            self._counts = ranked
            self._frames += 1
            self._updated = self._clock()

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            age = None if self._updated is None else self._clock() - self._updated
            return LiveSnapshot(self._fps, self._latency_ms, self._counts, self._frames, age)
