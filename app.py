"""VisionLab AI: real-time object detection and pose estimation from your browser webcam.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from visionlab import ui
from visionlab.filters import pencil_sketch
from visionlab.live import LiveStats
from visionlab.overlay import FPSMeter, draw_hud, summarize_counts
from visionlab.tasks import TASKS

st.set_page_config(
    page_title="VisionLab AI",
    page_icon="🎥",
    layout="wide",
    menu_items={"About": "VisionLab AI: real-time computer vision in the browser."},
)
st.html(ui.CSS)


@st.cache_resource(show_spinner="Loading model (the first launch downloads the weights)…")
def load_detector(weights: str):
    # Imported lazily so the sketch filter works without loading PyTorch.
    from visionlab.inference import Detector

    detector = Detector(weights)
    detector.warmup()
    return detector


@st.cache_data
def sample_images() -> dict[str, bytes]:
    """Demo photos bundled with Ultralytics, for visitors without a camera."""
    from ultralytics.utils import ASSETS

    return {
        "Street scene (bus.jpg)": (ASSETS / "bus.jpg").read_bytes(),
        "Two people (zidane.jpg)": (ASSETS / "zidane.jpg").read_bytes(),
    }


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("⚙️ Settings")
    task_key = st.radio(
        "Vision task",
        options=list(TASKS),
        format_func=lambda key: f"{TASKS[key].icon}  {TASKS[key].label}",
    )
    task = TASKS[task_key]
    st.caption(task.description)

    if task.uses_model:
        confidence = st.slider(
            "Minimum confidence", 0.10, 0.95, 0.40, 0.05,
            help="Hide predictions the model is less sure about than this.",
        )
        image_size = st.select_slider(
            "Inference resolution", options=[320, 416, 512, 640], value=416,
            help="Lower is faster (especially on CPU); higher finds smaller objects.",
        )
        sketch_detail, sketch_invert = 11, False
    else:
        confidence, image_size = 0.0, 0
        sketch_detail = st.slider(
            "Line detail", 3, 31, 11, 2,
            help="Adaptive-threshold block size. Smaller = finer, busier lines.",
        )
        sketch_invert = st.toggle("Invert (white lines on black)", value=False)

    show_hud = st.toggle("Stats overlay on video", value=True)
    st.divider()
    st.caption("💡 Video choppy? Drop the inference resolution to 320.")

detector = load_detector(task.weights) if task.uses_model else None
device_label = "CPU" if detector is None or detector.device == "cpu" else "GPU"


def process(image_bgr: np.ndarray):
    """Run the selected task on one frame.

    Returns (output_bgr, prediction_or_None, latency_ms).
    """
    if detector is None:
        start = time.perf_counter()
        output = pencil_sketch(image_bgr, detail=sketch_detail, invert=sketch_invert)
        return output, None, (time.perf_counter() - start) * 1000
    prediction = detector.predict(image_bgr, conf=confidence, imgsz=image_size)
    return prediction.annotated_bgr, prediction, prediction.latency_ms


# ----------------------------------------------------------------- header --
# The hero is a single element whatever the task, so the page structure stays
# stable and Streamlit doesn't reset the selected tab when the task changes.
st.html(ui.hero_html(task, device_label))

live_tab, image_tab, about_tab = st.tabs(["📹  Live webcam", "🖼️  Photo / upload", "ℹ️  How it works"])

# --------------------------------------------------------------- live tab --
with live_tab:
    if "fps_meter" not in st.session_state:
        st.session_state.fps_meter = FPSMeter()
        st.session_state.live_stats = LiveStats()
    fps_meter: FPSMeter = st.session_state.fps_meter
    live_stats: LiveStats = st.session_state.live_stats

    # Runs on a worker thread for every frame, so it must not call st.* APIs.
    # streamlit-webrtc swaps in the new closure on every rerun, which is how
    # sidebar changes reach a running stream.
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        output_bgr, prediction, latency_ms = process(image_bgr)
        fps = fps_meter.tick()
        labels = prediction.labels if prediction is not None else None
        live_stats.update(fps, latency_ms, labels)
        if show_hud:
            hud = [f"{fps:4.1f} FPS  |  {latency_ms:3.0f} ms"]
            if labels is not None:
                hud.append(summarize_counts(labels))
            draw_hud(output_bgr, hud)
        return av.VideoFrame.from_ndarray(output_bgr, format="bgr24")

    video_col, panel_col = st.columns([3, 2], gap="large")
    with video_col:
        ctx = webrtc_streamer(
            key="visionlab-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            # No native controls: a live stream has nothing to pause or seek, and
            # leaving browser fullscreen made the component collapse to a strip.
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "playsInline": True,
                "style": {
                    "width": "100%",
                    "maxHeight": "520px",
                    "objectFit": "contain",
                    "borderRadius": "14px",
                    "background": "#05070B",
                },
            },
            translations={"start": "▶ Start camera", "stop": "■ Stop", "select_device": "Select camera"},
            async_processing=True,
        )
        st.caption(
            "🔒 Frames are processed in memory and never saved or recorded. "
            "Video won't connect? Your network may block WebRTC. Try another "
            "network or use the **Photo / upload** tab."
        )

    playing = ctx.state.playing

    with panel_col:
        # Refreshes just this panel twice a second while the stream is live.
        @st.fragment(run_every=0.5 if playing else None)
        def live_panel():
            st.html(ui.live_panel_html(task, live_stats.snapshot(), playing))

        live_panel()
        st.html(ui.tips_html(task))

# -------------------------------------------------------------- image tab --
with image_tab:
    source = st.radio(
        "Image source", ["📸 Webcam snapshot", "📁 Upload an image", "✨ Try a sample"],
        horizontal=True, label_visibility="collapsed",
    )
    image_bytes: bytes | None = None
    if source.endswith("snapshot"):
        shot = st.camera_input("Take a photo to analyze")
        image_bytes = shot.getvalue() if shot else None
    elif source.endswith("image"):
        upload = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"])
        image_bytes = upload.getvalue() if upload else None
    else:
        samples = sample_images()
        image_bytes = samples[st.selectbox("Sample image", list(samples))]

    if image_bytes:
        # Decoded in memory; nothing is written to disk.
        image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Couldn't read that image. Please try another JPG or PNG.")
        else:
            with st.spinner("Running…"):
                output_bgr, prediction, latency_ms = process(image_bgr)

            before, after = st.columns(2)
            before.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
            after.image(
                cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB),
                caption=f"{task.icon} {task.label}",
                width="stretch",
            )

            m1, m2, m3 = st.columns(3)
            if prediction is not None:
                m1.metric("Detections", len(prediction.labels))
                m2.metric("Distinct classes", len(set(prediction.labels)))
            else:
                m1.metric("Detections", "–")
                m2.metric("Distinct classes", "–")
            m3.metric("Inference time" if prediction is not None else "Filter time", f"{latency_ms:.0f} ms")

            if prediction is not None and prediction.labels:
                best: dict[str, float] = defaultdict(float)
                counts: dict[str, int] = defaultdict(int)
                for label, conf in zip(prediction.labels, prediction.confidences, strict=True):
                    counts[label] += 1
                    best[label] = max(best[label], conf)
                rows = [
                    {"Object": label, "Count": counts[label], "Best confidence": best[label]}
                    for label in sorted(counts, key=lambda name: (-counts[name], name))
                ]
                st.dataframe(
                    rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Best confidence": st.column_config.ProgressColumn(
                            format="%.2f", min_value=0.0, max_value=1.0
                        )
                    },
                )
            elif prediction is not None:
                st.info("Nothing found above the confidence threshold. Try lowering it in the sidebar.")

            ok, encoded = cv2.imencode(".jpg", output_bgr)
            if ok:
                st.download_button(
                    "⬇️ Download result",
                    data=encoded.tobytes(),
                    file_name=f"visionlab_{task.key}.jpg",
                    mime="image/jpeg",
                    type="primary",
                )

# -------------------------------------------------------------- about tab --
with about_tab:
    st.subheader("From webcam to annotated video in four steps")
    st.html(ui.steps_html())

    models_col, privacy_col = st.columns([3, 2], gap="large")
    with models_col:
        st.markdown(
            """
#### Models

| Task | Model | Predicts |
|---|---|---|
| 🎯 Object detection | YOLO11n (COCO, 80 classes) | Boxes, classes, confidence |
| 🕺 Pose estimation | YOLO26n-pose (COCO keypoints) | 17 body keypoints per person |
| ✏️ Pencil sketch | OpenCV adaptive threshold | Line drawing, no neural network |

Both networks are pretrained "nano" variants, chosen for real-time speed on CPU.
"""
        )
    with privacy_col:
        st.markdown(
            """
#### Privacy

Frames go to the app server only for inference. They're held in memory and
never stored or recorded. Avoid pointing the camera at anything sensitive.
"""
        )

st.html(ui.FOOTER)
