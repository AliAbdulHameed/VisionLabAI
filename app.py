"""VisionLab AI: real-time object detection and pose estimation from your browser webcam.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

from collections import defaultdict

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from visionlab.filters import pencil_sketch
from visionlab.overlay import FPSMeter, draw_hud, summarize_counts
from visionlab.tasks import TASKS

st.set_page_config(
    page_title="VisionLab AI",
    page_icon="🎥",
    layout="wide",
    menu_items={"About": "VisionLab AI: real-time computer vision in the browser."},
)


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
        format_func=lambda key: TASKS[key].label,
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

    show_hud = st.toggle("Show FPS / latency overlay", value=True)

    st.divider()
    st.caption(
        "Built with Streamlit, WebRTC, OpenCV and Ultralytics YOLO. "
        "[Source on GitHub](https://github.com/AliAbdulHameed/VisionLabAI)"
    )

detector = load_detector(task.weights) if task.uses_model else None


def process(image_bgr: np.ndarray):
    """Run the selected task on one frame. Returns (output_bgr, prediction_or_None)."""
    if detector is None:
        return pencil_sketch(image_bgr, detail=sketch_detail, invert=sketch_invert), None
    prediction = detector.predict(image_bgr, conf=confidence, imgsz=image_size)
    return prediction.annotated_bgr, prediction


# ----------------------------------------------------------------- header --
st.title("🎥 VisionLab AI")
st.markdown(
    "Real-time **object detection**, **human pose estimation** and an OpenCV "
    "**sketch filter**, running on your own webcam, right in the browser."
)
# Always render this line: if the page structure changed between tasks,
# Streamlit would recreate the tabs below and jump back to the first one.
if detector is not None:
    st.caption(f"Model: `{task.weights}` · device: `{detector.device}`")
else:
    st.caption("Model: none (classic OpenCV) · device: `cpu`")

live_tab, image_tab, about_tab = st.tabs(["📹 Live webcam", "🖼️ Photo / upload", "ℹ️ How it works"])

# --------------------------------------------------------------- live tab --
with live_tab:
    st.markdown(
        "Click **START** and allow camera access. Each frame is sent to the app "
        "server, processed, and streamed back annotated. Switch tasks or move the "
        "sliders while the stream is running."
    )

    if "fps_meter" not in st.session_state:
        st.session_state.fps_meter = FPSMeter()
    fps_meter: FPSMeter = st.session_state.fps_meter

    # Runs on a worker thread for every frame, so it must not call st.* APIs.
    # streamlit-webrtc swaps in the new closure on every rerun, which is how
    # sidebar changes reach a running stream.
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        output_bgr, prediction = process(image_bgr)
        if show_hud:
            fps = fps_meter.tick()
            if prediction is None:
                hud = [f"{fps:4.1f} FPS"]
            else:
                hud = [f"{fps:4.1f} FPS  |  {prediction.latency_ms:3.0f} ms inference",
                       summarize_counts(prediction.labels)]
            draw_hud(output_bgr, hud)
        return av.VideoFrame.from_ndarray(output_bgr, format="bgr24")

    webrtc_streamer(
        key="visionlab-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )
    st.caption(
        "🔒 Frames are processed in memory and never saved or recorded. "
        "If the video doesn't connect, your network may block WebRTC: try "
        "another network or use the **Photo / upload** tab."
    )

# -------------------------------------------------------------- image tab --
with image_tab:
    source = st.radio(
        "Image source", ["Webcam snapshot", "Upload an image", "Try a sample"],
        horizontal=True, label_visibility="collapsed",
    )
    image_bytes: bytes | None = None
    if source == "Webcam snapshot":
        shot = st.camera_input("Take a photo to analyze")
        image_bytes = shot.getvalue() if shot else None
    elif source == "Upload an image":
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
                output_bgr, prediction = process(image_bgr)

            before, after = st.columns(2)
            before.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
            after.image(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), caption=task.label, width="stretch")

            if prediction is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Detections", len(prediction.labels))
                m2.metric("Distinct classes", len(set(prediction.labels)))
                m3.metric("Inference time", f"{prediction.latency_ms:.0f} ms")

                if prediction.labels:
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
                else:
                    st.info("Nothing found above the confidence threshold. Try lowering it in the sidebar.")

            ok, encoded = cv2.imencode(".jpg", output_bgr)
            if ok:
                st.download_button(
                    "⬇️ Download result",
                    data=encoded.tobytes(),
                    file_name=f"visionlab_{task.key}.jpg",
                    mime="image/jpeg",
                )

# -------------------------------------------------------------- about tab --
with about_tab:
    st.markdown(
        """
### Pipeline

```
Browser webcam ──WebRTC──▶ Streamlit server ──▶ YOLO / OpenCV ──▶ annotated frame ──WebRTC──▶ Browser
```

1. **Capture:** the browser grabs your camera with `getUserMedia` and streams it over WebRTC
   (via [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)).
2. **Inference:** each frame is decoded with PyAV into a NumPy array and passed to the selected model.
   A single cached model is shared across visitors, and a lock keeps calls into it thread-safe.
3. **Render:** Ultralytics draws boxes/skeletons, a HUD adds FPS, latency and object counts,
   and the frame is streamed back.

### Models

| Task | Model | What it predicts |
|---|---|---|
| Object detection | YOLO11n (COCO, 80 classes) | Bounding box + class + confidence |
| Pose estimation | YOLO26n-pose (COCO keypoints) | 17 body keypoints per person |
| Pencil sketch | OpenCV adaptive threshold | Line drawing, no neural network |

Both networks are pretrained "nano" variants, chosen for real-time speed on CPU.

### Privacy

Frames go to the app server only for inference. They are held in memory and
never stored or recorded. Avoid pointing the camera at anything sensitive.
"""
    )
