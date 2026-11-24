from pathlib import Path

import cv2
import numpy as np
import pytest

from visionlab.draw import draw_detections
from visionlab.inference import (
    COCO_NAMES,
    Detections,
    Detector,
    letterbox,
    nms,
    postprocess,
    scale_to_original,
    to_tensor,
)
from visionlab.tasks import TASKS

SAMPLES = Path("assets/samples")


def test_letterbox_pads_to_multiple_of_32():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, (new_w, new_h, left, top) = letterbox(image, 416)
    assert padded.shape == (320, 416, 3)  # 640x480 scaled to 416x312, padded to 416x320
    assert (new_w, new_h, left, top) == (416, 312, 0, 4)
    assert (padded[:4] == 114).all() and (padded[-4:] == 114).all()


def test_to_tensor_layout_and_range():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    tensor = to_tensor(image)
    assert tensor.shape == (1, 3, 2, 3) and tensor.dtype == np.float32
    assert tensor[0, 2].max() == 1.0 and tensor[0, 0].max() == 0.0  # blue lands in the last (RGB) channel


def test_nms_suppresses_overlaps_only():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]
    assert nms(boxes, scores, 0.95).tolist() == [0, 1, 2]


def test_postprocess_filters_by_confidence_and_class_aware_nms():
    # 3 anchors, 2 classes: two overlapping boxes of different classes and one low-confidence box.
    raw = np.array(
        [
            # cx, cy, w, h, class0, class1
            [50, 50, 20, 20, 0.9, 0.1],
            [51, 51, 20, 20, 0.1, 0.8],
            [10, 10, 5, 5, 0.05, 0.1],
        ],
        dtype=np.float32,
    ).T[None]
    boxes, scores, classes, extra = postprocess(raw, num_classes=2, conf=0.25)
    assert classes.tolist() == [0, 1]  # overlap kept: different classes
    np.testing.assert_allclose(boxes[0], [40, 40, 60, 60])
    assert extra.shape == (2, 0)


def test_scale_to_original_inverts_letterbox():
    info = (416, 312, 0, 4)  # from a 640x480 frame at imgsz 416
    letterboxed = np.array([[208.0, 160.0]])  # centre of the padded 416x320 input
    np.testing.assert_allclose(scale_to_original(letterboxed, info, (480, 640)), [[320.0, 240.0]])


@pytest.fixture(scope="module")
def detector():
    return Detector(TASKS["detect"].model_dir, "detect", threads=2)


@pytest.fixture(scope="module")
def pose_detector():
    return Detector(TASKS["pose"].model_dir, "pose", threads=2)


def test_detects_bus_and_people(detector):
    image = cv2.imread(str(SAMPLES / "bus.jpg"))
    prediction = detector.predict(image, conf=0.4, imgsz=416)
    assert "bus" in prediction.labels
    assert prediction.labels.count("person") >= 3
    assert prediction.annotated_bgr.shape == image.shape
    assert not np.array_equal(prediction.annotated_bgr, image)  # boxes were drawn


def test_pose_finds_two_skeletons(pose_detector):
    image = cv2.imread(str(SAMPLES / "zidane.jpg"))
    detections = pose_detector.detect(image, conf=0.4, imgsz=416)
    assert len(detections) == 2
    assert detections.keypoints.shape == (2, 17, 3)
    h, w = image.shape[:2]
    assert (detections.keypoints[..., 0] <= w).all() and (detections.keypoints[..., 1] <= h).all()


def test_blank_frame_has_no_detections(detector):
    assert len(detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), conf=0.25, imgsz=320)) == 0


def test_draw_detections_handles_empty_and_pose():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    empty = Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))
    assert draw_detections(image.copy(), empty, COCO_NAMES).sum() == 0
    keypoints = np.tile(np.array([[50, 50, 0.9]], dtype=np.float32), (17, 1))[None]
    box, score = np.array([[10, 10, 90, 90]], np.float32), np.array([0.9], np.float32)
    one = Detections(box, score, np.array([0]), keypoints)
    assert draw_detections(image.copy(), one, COCO_NAMES).sum() > 0
