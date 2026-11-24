<div align="center">

# 🎥 VisionLab AI

**Real-time object detection and human pose estimation, running on your own webcam, in the browser.**

<!-- After deploying, add your live demo badge here, e.g.:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP.streamlit.app)
-->

[![CI](https://github.com/AliAbdulHameed/VisionLabAI/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAbdulHameed/VisionLabAI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11%20%7C%20YOLO26-111F68)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<img src="docs/images/detection.jpg" alt="VisionLab AI running YOLO11 object detection on a street scene" width="900">

</div>

## Overview

VisionLab AI is a Streamlit web app that streams a visitor's webcam to a Python backend over **WebRTC**, runs a computer vision model on every frame, and streams the annotated video back in real time. No install is needed on the visitor's side: open the page, click **Start**, and allow camera access.

The project started as a set of Jupyter experiments with OpenCV video processing, MoveNet and YOLO pose estimation, and local webcam detection. I turned the techniques that worked best into an interactive, deployable app that anyone can use.

## Features

| | |
|---|---|
| 📹 **Live webcam inference** | Browser camera → WebRTC → model → annotated stream, side by side with a live dashboard of FPS, inference latency and the objects in frame |
| 🎯 **Object detection** | YOLO11n, pretrained on COCO (80 classes: people, phones, laptops, cups, TVs…) |
| 🕺 **Pose estimation** | YOLO26n-pose: 17 body keypoints and a skeleton for every person in the frame |
| ✏️ **Pencil-sketch filter** | Classic OpenCV pipeline (grayscale → median blur → adaptive threshold), no neural network |
| 🖼️ **Photo mode** | Analyze a webcam snapshot, an uploaded image or a built-in sample. See a per-class detection table with confidence and download the annotated result |
| 🎛️ **Live controls** | Switch tasks and tune the confidence threshold and inference resolution *while the stream is running*, with per-task tips for what to try |
| 🎞️ **Smooth video on a CPU** | The model runs off the video thread on the newest frame, so the stream plays at camera speed while detections refresh as fast as the model allows |
| ⚡ **Optimized inference** | OpenVINO runs the models 3–4× faster than PyTorch on CPU, with threads sized to the container's CPU quota; a CUDA GPU is used automatically when present |
| 🔒 **Privacy-minded** | Frames are processed in memory and never written to disk or recorded |

<table>
<tr>
<td><img src="docs/images/pose.jpg" alt="YOLO26 pose estimation"></td>
<td><img src="docs/images/sketch.jpg" alt="OpenCV pencil sketch filter"></td>
</tr>
<tr>
<td align="center"><b>Pose estimation (YOLO26n-pose)</b></td>
<td align="center"><b>Pencil sketch (OpenCV)</b></td>
</tr>
</table>

## How it works

```mermaid
flowchart LR
    A[Browser webcam<br/>getUserMedia] -- WebRTC video --> B[streamlit-webrtc<br/>frame callback]
    B -- PyAV → NumPy BGR --> C{Selected task}
    C -- detect --> D[YOLO11n]
    C -- pose --> E[YOLO26n-pose]
    C -- sketch --> F[OpenCV adaptive threshold]
    D & E & F --> G[Draw boxes / skeleton<br/>+ FPS & latency HUD]
    G -- WebRTC video --> H[Browser display]
```

- **Streaming:** [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) handles the peer connection. Each incoming `av.VideoFrame` goes to a callback on a worker thread, and the callback returns the processed frame.
- **Non-blocking video:** the frame callback never waits for the model. It hands a copy of the newest frame to a background worker and immediately redraws the latest finished detections onto the current frame. Frames that arrive while the model is busy replace the pending one, so there's no backlog and boxes are at most one inference behind.
- **OpenVINO on CPU:** on first load each model is exported to OpenVINO IR with dynamic input shapes (about 2 s) and reused afterwards. If OpenVINO is unavailable or the export fails, the app falls back to PyTorch.
- **CPU-quota aware:** cloud containers often see every host core but may only use a fraction of them. The app reads the cgroup CPU quota and caps inference threads to fit, leaving a core for video encoding. Otherwise the runtime spawns a thread per visible core, exhausts the quota in a burst and gets frozen, which shows up as stutter.
- **Shared, thread-safe models:** each model loads once with `st.cache_resource`, warms up on a dummy frame, and is shared by all visitors. A lock serializes calls into the model, because Ultralytics predictors aren't safe to call from several threads at once.
- **Live settings:** Streamlit reruns the script when a slider changes, and `streamlit-webrtc` swaps in the new frame callback. The running stream picks up the new settings without reconnecting.
- **Live dashboard:** the frame callback publishes FPS, latency and per-class counts to a lock-protected `LiveStats` object. A Streamlit fragment re-renders only the stats panel twice a second, so the rest of the page and the video stream are untouched.
- **Separation of concerns:** UI code lives in `app.py`. The vision logic in `visionlab/` is plain Python and is unit tested without a browser or GPU.

## Project structure

```
VisionLabAI/
├── app.py                  # Streamlit app: sidebar, live/photo/about tabs
├── visionlab/
│   ├── tasks.py            # Task registry (labels, descriptions, weights)
│   ├── inference.py        # YOLO wrapper: OpenVINO export, CPU-quota-aware threads
│   ├── background.py       # Non-blocking, latest-frame-wins inference worker
│   ├── filters.py          # OpenCV pencil-sketch filter
│   ├── overlay.py          # FPS meter, label summaries, HUD drawing
│   ├── live.py             # Thread-safe stats hand-off from video thread to UI
│   ├── ui.py               # HTML components (hero, live panel, cards)
│   └── style.css           # Custom theme and layout
├── tests/                  # pytest unit tests (run in GitHub Actions)
├── .streamlit/config.toml  # Theme
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # + pytest, ruff
└── packages.txt            # System libraries for Streamlit Community Cloud
```

## Run it locally

**Requirements:** Python 3.10–3.12 and a webcam. Model weights (~6 MB each) download automatically on first use.

```bash
git clone https://github.com/AliAbdulHameed/VisionLabAI.git
cd VisionLabAI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

<details>
<summary>Using Anaconda instead</summary>

```bash
conda create -n visionlab python=3.11 -y
conda activate visionlab
pip install -r requirements.txt
streamlit run app.py
```
</details>

Open the URL Streamlit prints (usually <http://localhost:8501>), go to **Live webcam**, click **START** and allow camera access.

> **Tip:** If you have an NVIDIA GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before installing the requirements. The app detects it automatically and shows `device: 0` under the title.

### Tests and linting

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Deploy

The app is ready for [Streamlit Community Cloud](https://streamlit.io/cloud) (free):

1. Sign in with GitHub, click **Create app**, and select this repository with `app.py` as the entry point.
2. Under **Advanced settings**, choose Python 3.11 or 3.12.
3. Deploy. `requirements.txt` and `packages.txt` are installed automatically.

**WebRTC and TURN:** Live video needs a direct peer connection between the visitor and the server. Many cloud hosts and corporate/mobile networks block that, so the **Photo / upload** tab is the reliable fallback. For dependable live video, add a TURN relay. Get free credentials from [Cloudflare Realtime TURN](https://developers.cloudflare.com/realtime/turn/), then add them in your app's **Settings → Secrets**:

```toml
CLOUDFLARE_TURN_KEY_ID = "..."
CLOUDFLARE_TURN_KEY_API_TOKEN = "..."
```

`streamlit-webrtc` detects these environment variables and configures the ICE servers automatically. Never commit keys to the repository. See the [streamlit-webrtc deployment guide](https://github.com/whitphx/streamlit-webrtc#network-configuration) for details.

**Performance:** Free hosting is CPU-only and shared. Measured locally inside a container limited to a 1-CPU quota (YOLO11n at 416 px, 20 fps test camera):

| | Blocking + PyTorch | Non-blocking + OpenVINO |
|---|---|---|
| Inference latency | 87–426 ms | 18–22 ms |
| Frames shown in the browser | 10–12 fps | 14–16 fps |

With more CPU or a 30 fps webcam the video rate rises further. The remaining ceiling on a single CPU is WebRTC video encoding and decoding, not the model.

## Roadmap

- [ ] Multi-object tracking with persistent IDs (ByteTrack)
- [ ] Video-file upload with downloadable annotated output
- [ ] Rep counting / joint-angle measurement from pose keypoints
- [ ] Class filter (e.g. detect only `person` and `cell phone`)
- [ ] ONNX / OpenVINO export for faster CPU inference

## Acknowledgements and licensing

- Models and inference: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The YOLO11 and YOLO26 weights are pretrained by Ultralytics on the [COCO](https://cocodataset.org/) dataset. This project integrates and deploys them; it does not retrain them.
- Streaming: [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) by Yuichiro Tachibana.
- Sample images (`bus.jpg`, `zidane.jpg`) ship with the Ultralytics package.

The code in this repository is released under the [MIT License](LICENSE). Ultralytics YOLO is licensed under [AGPL-3.0](https://www.ultralytics.com/license), which applies to any deployment that uses it. Review those terms before using this project commercially.
