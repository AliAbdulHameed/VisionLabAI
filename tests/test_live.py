import threading

from visionlab.live import LiveStats


class ManualClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_empty_snapshot():
    snap = LiveStats().snapshot()
    assert snap.frames == 0
    assert snap.age_s is None
    assert not snap.is_fresh()
    assert snap.total_objects == 0


def test_update_ranks_counts_and_tracks_age():
    clock = ManualClock()
    stats = LiveStats(clock=clock)
    stats.update(fps=24.0, latency_ms=31.0, labels=["cup", "person", "person"])
    clock.now += 0.5
    snap = stats.snapshot()
    assert snap.counts == (("person", 2), ("cup", 1))
    assert snap.total_objects == 3
    assert snap.fps == 24.0 and snap.latency_ms == 31.0
    assert snap.is_fresh()
    clock.now += 5
    assert not stats.snapshot().is_fresh()


def test_labels_none_means_no_objects():
    stats = LiveStats()
    stats.update(fps=30.0, latency_ms=2.0, labels=None)
    assert stats.snapshot().counts == ()


def test_concurrent_updates_count_every_frame():
    stats = LiveStats()

    def worker():
        for _ in range(500):
            stats.update(10.0, 5.0, ["person"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stats.snapshot().frames == 2000
