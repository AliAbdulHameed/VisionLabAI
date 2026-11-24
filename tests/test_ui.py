from visionlab import ui
from visionlab.live import LiveSnapshot
from visionlab.tasks import TASKS


def _snap(age=0.1, counts=(("person", 2),)):
    return LiveSnapshot(fps=29.5, latency_ms=22.0, counts=counts, frames=10, age_s=age)


def test_idle_panel_prompts_to_start():
    html = ui.live_panel_html(TASKS["detect"], _snap(age=None), playing=False)
    assert "Start camera" in html and "Camera off" in html


def test_live_panel_shows_stats_and_detections():
    html = ui.live_panel_html(TASKS["detect"], _snap(), playing=True)
    assert "29.5" in html and "22<small> ms</small>" in html and "person<b>2</b>" in html
    assert "#042aff" in html  # legend swatch matches the box colour drawn for people


def test_stale_stream_shows_connecting():
    html = ui.live_panel_html(TASKS["pose"], _snap(age=10), playing=True)
    assert "Connecting" in html


def test_labels_are_html_escaped():
    html = ui.live_panel_html(TASKS["detect"], _snap(counts=(("<script>", 1),)), playing=True)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_hero_and_tips_render():
    assert "VisionLab AI" in ui.hero_html(TASKS["sketch"], "CPU")
    assert all(tip in ui.tips_html(TASKS["pose"]) for tip in TASKS["pose"].tips)


def test_swatch_matches_drawn_box_colours():
    assert ui.swatch("person") == "#042aff"
    assert ui.swatch("bicycle") == "#0bdbeb"
    assert ui.swatch("not-a-class") == "#042aff"
