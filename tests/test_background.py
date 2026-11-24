import threading
import time

import numpy as np

from visionlab.background import BackgroundInference


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def frame(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_result_becomes_available_for_matching_tag():
    worker = BackgroundInference()
    worker.submit(frame(7), lambda f: int(f[0, 0, 0]), tag="detect")
    assert wait_for(lambda: worker.latest("detect") == 7)
    assert worker.latest("pose") is None  # results never leak across tasks


def test_busy_worker_skips_to_newest_frame():
    release = threading.Event()
    seen = []

    def slow_job(f):
        seen.append(int(f[0, 0, 0]))
        release.wait(1.0)
        return seen[-1]

    worker = BackgroundInference()
    worker.submit(frame(1), slow_job, "t")
    assert wait_for(lambda: seen == [1])
    for value in range(2, 10):  # arrive while the model is busy
        worker.submit(frame(value), slow_job, "t")
    release.set()
    assert wait_for(lambda: worker.latest("t") == 9)
    assert seen == [1, 9]  # frames 2-8 were dropped, no backlog


def test_job_errors_do_not_kill_the_worker():
    worker = BackgroundInference()
    worker.submit(frame(0), lambda f: 1 / 0, "t")
    worker.submit(frame(5), lambda f: "ok", "t")
    assert wait_for(lambda: worker.latest("t") == "ok")


def test_thread_exits_when_idle_and_restarts_on_demand():
    worker = BackgroundInference(idle_timeout_s=0.05)
    worker.submit(frame(1), lambda f: "a", "t")
    assert wait_for(lambda: worker._thread is None)
    worker.submit(frame(2), lambda f: "b", "t")
    assert wait_for(lambda: worker.latest("t") == "b")
