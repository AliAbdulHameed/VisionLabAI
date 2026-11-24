import numpy as np
import pytest

from visionlab.overlay import FPSMeter, draw_hud, summarize_counts


class FakeClock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def test_fps_meter_converges_to_frame_rate():
    meter = FPSMeter(clock=FakeClock(step=0.05))  # 20 FPS
    for _ in range(50):
        fps = meter.tick()
    assert fps == pytest.approx(20.0)


def test_fps_meter_first_tick_is_zero():
    assert FPSMeter(clock=FakeClock(0.1)).tick() == 0.0


def test_fps_meter_rejects_bad_smoothing():
    with pytest.raises(ValueError):
        FPSMeter(smoothing=1.0)


def test_summarize_counts_orders_by_frequency_then_name():
    labels = ["tv", "person", "cup", "person", "cup", "person"]
    assert summarize_counts(labels) == "3 person · 2 cup · 1 tv"


def test_summarize_counts_truncates():
    assert summarize_counts(["a", "b", "c"], limit=2) == "1 a · 1 b · +1 more"


def test_summarize_counts_empty():
    assert summarize_counts([]) == "nothing detected"


def test_draw_hud_darkens_panel_and_keeps_shape():
    image = np.full((100, 300, 3), 200, dtype=np.uint8)
    out = draw_hud(image.copy(), ["12.0 FPS", "1 person"])
    assert out.shape == image.shape
    assert out[12, 12].mean() < 200       # inside the panel
    assert (out[90, 290] == 200).all()    # untouched corner


def test_draw_hud_handles_tiny_frames():
    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    draw_hud(tiny, ["a long line of text that does not fit"])


def test_fps_meter_resets_after_a_pause():
    clock = FakeClock(step=0.05)
    meter = FPSMeter(clock=clock)
    for _ in range(20):
        meter.tick()
    clock.step = 10.0  # stream paused
    assert meter.tick() == 0.0
    clock.step = 0.1  # resumed at 10 FPS
    assert meter.tick() == pytest.approx(10.0)
