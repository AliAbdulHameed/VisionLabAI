"""The computer vision tasks the app offers.

Kept free of heavy imports so the UI and tests can read task metadata
without loading PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    description: str
    weights: str | None = None  # None means the task is pure OpenCV, no model

    @property
    def uses_model(self) -> bool:
        return self.weights is not None


TASKS: dict[str, Task] = {
    task.key: task
    for task in (
        Task(
            key="detect",
            label="Object detection · YOLO11n",
            description="Finds and labels the 80 COCO object classes (people, phones, cups, TVs, ...).",
            weights="yolo11n.pt",
        ),
        Task(
            key="pose",
            label="Pose estimation · YOLO26n-pose",
            description="Tracks 17 body keypoints per person and draws a skeleton.",
            weights="yolo26n-pose.pt",
        ),
        Task(
            key="sketch",
            label="Pencil sketch · OpenCV",
            description=(
                "Classic computer vision: adaptive thresholding turns each frame "
                "into a line drawing. No neural network."
            ),
        ),
    )
}
