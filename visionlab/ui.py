"""Styling and HTML snippets for the Streamlit UI."""

from __future__ import annotations

from html import escape
from pathlib import Path

from visionlab.live import LiveSnapshot
from visionlab.tasks import Task

CSS = "<style>" + (Path(__file__).with_name("style.css")).read_text(encoding="utf-8") + "</style>"

STACK = ("YOLO11", "YOLO26-pose", "PyTorch", "OpenCV", "WebRTC", "Streamlit")


def hero_html(task: Task, device: str) -> str:
    chips = "".join(f'<span class="vl-chip">{escape(name)}</span>' for name in STACK)
    return f"""
<div class="vl-hero">
  <div class="vl-status"><span class="vl-dot"></span>{escape(task.model)} · {escape(device)}</div>
  <h1 class="vl-title">VisionLab AI</h1>
  <p class="vl-tagline">Real-time <b>object detection</b>, <b>pose estimation</b> and an OpenCV
  <b>sketch filter</b>, running on your own webcam, right in the browser.</p>
  <div class="vl-chips">{chips}</div>
</div>
"""


def task_header_html(task: Task) -> str:
    return f"""
<div class="vl-card-head">
  <div class="vl-icon">{task.icon}</div>
  <div>
    <div class="vl-card-title">{escape(task.name)}</div>
    <div class="vl-card-sub">{escape(task.model)}</div>
  </div>
</div>
"""


def live_panel_html(task: Task, snap: LiveSnapshot, playing: bool) -> str:
    """Live stats card: FPS, latency, object count and what's in frame."""
    live = playing and snap.is_fresh()

    def stat(value: str, unit: str, label: str) -> str:
        unit_html = f"<small>{unit}</small>" if unit else ""
        return f'<div class="vl-stat"><div class="vl-stat-val">{value}{unit_html}</div>' \
               f'<div class="vl-stat-lbl">{label}</div></div>'

    if live:
        objects = str(snap.total_objects) if task.uses_model else "–"
        stats = stat(f"{snap.fps:.1f}", "", "Frames / sec") + \
            stat(f"{snap.latency_ms:.0f}", "ms", "Inference" if task.uses_model else "Filter time") + \
            stat(objects, "", "People" if task.key == "pose" else "Objects")
    else:
        stats = stat("–", "", "Frames / sec") + stat("–", "", "Latency") + stat("–", "", "Objects")

    if not playing:
        body = '<div class="vl-idle">Press <b>▶ Start camera</b> and allow access to go live.</div>'
    elif not live:
        body = '<div class="vl-idle">Connecting to the stream…</div>'
    elif not task.uses_model:
        body = '<div class="vl-idle">Drawing every frame with adaptive thresholding.</div>'
    elif snap.counts:
        body = '<div class="vl-chips">' + "".join(
            f'<span class="vl-det">{escape(label)}<b>×{count}</b></span>' for label, count in snap.counts
        ) + "</div>"
    else:
        body = '<div class="vl-idle">Nothing above the confidence threshold yet.</div>'

    dot = "vl-dot" if live else "vl-dot off"
    status = "Live" if live else "Idle"
    return f"""
<div class="vl-card">
  {task_header_html(task)}
  <div class="vl-section"><span class="{dot}"></span>{status}</div>
  <div class="vl-stats">{stats}</div>
  <div class="vl-section">In frame now</div>
  {body}
</div>
"""


def tips_html(task: Task) -> str:
    items = "".join(f"<li>{escape(tip)}</li>" for tip in task.tips)
    return f'<div class="vl-card"><div class="vl-section" style="margin-top:0">Try this</div>' \
           f'<ul class="vl-tips">{items}</ul></div>'


STEPS = (
    ("Capture", "Your browser grabs the camera with getUserMedia and streams it to the server over WebRTC."),
    ("Decode", "Each frame arrives in a Python callback and is decoded with PyAV into a NumPy array."),
    ("Infer", "One cached YOLO model, shared across visitors behind a lock, predicts boxes or keypoints."),
    ("Render", "Boxes, skeletons and a stats overlay are drawn, then the frame streams straight back."),
)


def steps_html() -> str:
    cards = "".join(
        f'<div class="vl-step"><div class="vl-step-n">{i}</div><h4>{title}</h4><p>{text}</p></div>'
        for i, (title, text) in enumerate(STEPS, start=1)
    )
    return f'<div class="vl-steps">{cards}</div>'


FOOTER = """
<div class="vl-footer">
  VisionLab AI · Built with Streamlit, WebRTC, OpenCV and Ultralytics YOLO ·
  <a href="https://github.com/AliAbdulHameed/VisionLabAI" target="_blank">Source on GitHub</a>
</div>
"""
