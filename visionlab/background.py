"""Run slow per-frame work off the video thread.

The live video callback must return quickly, or the stream stutters. Instead
of running the model on every frame, the callback hands its newest frame to a
background worker and immediately draws the most recent finished result onto
the current frame. The video plays at camera speed; detections refresh as fast
as the model can go. Frames that arrive while the model is busy simply replace
the pending one, so the worker always processes the newest frame and never
builds up a backlog.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

Job = Callable[[np.ndarray], Any]


class BackgroundInference:
    def __init__(self, idle_timeout_s: float = 5.0):
        self._idle_timeout_s = idle_timeout_s
        self._cond = threading.Condition()
        self._pending: tuple[np.ndarray, Job, Hashable] | None = None
        self._latest: tuple[Hashable, Any] | None = None
        self._thread: threading.Thread | None = None
        self.completed = 0

    def submit(self, frame: np.ndarray, job: Job, tag: Hashable) -> None:
        """Queue ``job(frame)``, replacing any frame still waiting to be processed.

        ``tag`` identifies what produced the result (e.g. the task), so results
        from a previous setting are never drawn after the user switches.
        """
        with self._cond:
            self._pending = (frame, job, tag)
            self._cond.notify()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="visionlab-inference", daemon=True)
                self._thread.start()

    def latest(self, tag: Hashable) -> Any | None:
        """The newest finished result for ``tag``, or None if there isn't one yet."""
        with self._cond:
            if self._latest is not None and self._latest[0] == tag:
                return self._latest[1]
            return None

    def _run(self) -> None:
        while True:
            with self._cond:
                has_work = self._cond.wait_for(lambda: self._pending is not None, self._idle_timeout_s)
                if not has_work:
                    # Stream stopped: let the thread end; submit() starts a new one.
                    self._thread = None
                    return
                frame, job, tag = self._pending
                self._pending = None
            try:
                result = job(frame)
            except Exception:
                LOGGER.exception("Background inference failed")
                continue
            with self._cond:
                self._latest = (tag, result)
                self.completed += 1
