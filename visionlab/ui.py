"""Styling and HTML snippets for the Streamlit UI."""

from __future__ import annotations

from html import escape
from pathlib import Path

from visionlab.draw import PALETTE_BGR
from visionlab.inference import COCO_NAMES
from visionlab.live import LiveSnapshot
from visionlab.tasks import Task

CSS = "<style>" + (Path(__file__).with_name("style.css")).read_text(encoding="utf-8") + "</style>"


def hero_html(task: Task, device: str) -> str:
    return f"""
<div class="vl-hero">
  <div class="vl-box"><span class="vl-tag">project 0.99</span><h1 class="vl-title">VisionLab AI</h1></div>
  <p class="vl-lede">Turn on your webcam and watch a neural network name what it sees, follow your pose,
  or redraw the room as a pencil sketch, live in your browser.</p>
  <p class="vl-meta">{escape(task.model)} running on {escape(device)}.
  Frames are processed in memory and never stored.</p>
</div>
"""


def swatch(label: str) -> str:
    """Hex colour of the box drawn for ``label``, so the legend matches the video."""
    cls = COCO_NAMES.index(label) if label in COCO_NAMES else 0
    b, g, r = PALETTE_BGR[cls % len(PALETTE_BGR)]
    return f"#{r:02x}{g:02x}{b:02x}"


def live_panel_html(task: Task, snap: LiveSnapshot, playing: bool) -> str:
    """Live readout: frame rate, inference time, and what's in view with box colours."""
    live = playing and snap.is_fresh()

    def fig(value: str, label: str) -> str:
        return f'<div class="vl-fig"><b>{value}</b><span>{label}</span></div>'

    noun = "people" if task.key == "pose" else "objects"
    if live:
        figs = fig(f"{snap.fps:.1f}", "frames per second") + fig(
            f"{snap.latency_ms:.0f}<small> ms</small>", "per inference" if task.uses_model else "per frame"
        ) + fig(str(snap.total_objects) if task.uses_model else "–", f"{noun} in view")
    else:
        figs = fig("–", "frames per second") + fig("–", "per inference") + fig("–", f"{noun} in view")

    if not playing:
        state, body = "Camera off", '<p class="vl-note">Select <b>Start camera</b> and allow access.</p>'
    elif not live:
        state, body = "Connecting…", '<p class="vl-note">Waiting for the first frames.</p>'
    elif not task.uses_model:
        state, body = "Live", '<p class="vl-note">Every frame is redrawn with adaptive thresholding.</p>'
    elif snap.counts:
        state = "Live"
        body = '<ul class="vl-legend">' + "".join(
            f'<li><i style="background:{swatch(label)}"></i>{escape(label)}<b>{count}</b></li>'
            for label, count in snap.counts
        ) + "</ul>"
    else:
        state, body = "Live", '<p class="vl-note">Nothing in view above the confidence threshold.</p>'

    return f"""
<div class="vl-readout">
  <h3>{escape(task.name)}</h3>
  <p class="vl-model">{escape(task.model)}</p>
  <div class="vl-state"><span class="vl-rec{' on' if live else ''}"></span>{state}</div>
  <div class="vl-figs">{figs}</div>
  {body}
</div>
"""


def tips_html(task: Task) -> str:
    items = "".join(f"<li>{escape(tip)}</li>" for tip in task.tips)
    return f'<div class="vl-tips"><h4>Things to try</h4><ul>{items}</ul></div>'


STEPS = (
    ("Capture", "Your browser grabs the camera with getUserMedia and streams it to the server over WebRTC."),
    ("Decode", "Each frame arrives in a Python callback and is decoded with PyAV into a NumPy array."),
    ("Infer", "A background worker runs YOLO via OpenVINO on the newest frame, so video never waits."),
    ("Render", "The latest boxes or skeletons are drawn onto every frame, which streams straight back."),
)


def steps_html() -> str:
    cards = "".join(
        f'<div class="vl-step"><div class="vl-step-n">{i}</div><h4>{title}</h4><p>{text}</p></div>'
        for i, (title, text) in enumerate(STEPS, start=1)
    )
    return f'<div class="vl-steps">{cards}</div>'


FOOTER = """
<div class="vl-footer">
  VisionLab AI is open source.
  <a href="https://github.com/AliAbdulHameed/VisionLabAI" target="_blank">Read the code on GitHub</a>.
  Models by Ultralytics, running on OpenVINO.
</div>
"""
