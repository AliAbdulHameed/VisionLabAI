"""The computer vision tasks the app offers.

Kept free of heavy imports so the UI and tests can read task metadata
without loading PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    model: str
    icon: str
    description: str
    tips: tuple[str, ...] = ()
    weights: str | None = None  # None means the task is pure OpenCV, no model

    @property
    def label(self) -> str:
        return f"{self.name} · {self.model}"

    @property
    def uses_model(self) -> bool:
        return self.weights is not None


TASKS: dict[str, Task] = {
    task.key: task
    for task in (
        Task(
            key="detect",
            name="Object detection",
            model="YOLO11n",
            icon="🎯",
            description="Finds and labels the 80 COCO object classes (people, phones, cups, TVs, ...).",
            tips=(
                "Hold up your phone, a cup, a bottle or a book",
                "Lower the confidence slider to see less certain guesses",
                "Raise the resolution to catch small objects in the background",
            ),
            weights="yolo11n.pt",
        ),
        Task(
            key="pose",
            name="Pose estimation",
            model="YOLO26n-pose",
            icon="🕺",
            description="Tracks 17 body keypoints per person and draws a skeleton.",
            tips=(
                "Lean back so your shoulders and arms are in frame",
                "Wave or raise a hand and watch the skeleton follow",
                "Grab a friend: every person gets their own skeleton",
            ),
            weights="yolo26n-pose.pt",
        ),
        Task(
            key="sketch",
            name="Pencil sketch",
            model="OpenCV",
            icon="✏️",
            description=(
                "Classic computer vision: adaptive thresholding turns each frame "
                "into a line drawing. No neural network."
            ),
            tips=(
                "Lower the line detail for a finer, busier drawing",
                "Toggle invert for a chalkboard look",
                "Good, even lighting gives the cleanest lines",
            ),
        ),
    )
}
