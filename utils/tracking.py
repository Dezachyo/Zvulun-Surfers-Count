# utils/tracking.py

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sahi.predict import get_sliced_prediction
import cv2


def sahi_detect_frame(
    frame_bgr: np.ndarray,
    sahi_detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    conf_th=0.5,
):
    """
    Returns detections in a YOLO-like format:
      boxes_xyxy: (N,4) float
      scores: (N,) float
      class_ids: (N,) int
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        image=frame_rgb,
        detection_model=sahi_detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        postprocess_match_threshold=0.5,  # tweak later
        postprocess_type="GREEDYNMM",
        verbose=0,
    )

    preds = result.object_prediction_list

    boxes = []
    scores = []
    class_ids = []

    for p in preds:
        if p.score.value < conf_th:
            continue

        # SAHI bbox can be accessed like this:
        # p.bbox.to_xyxy() returns [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        boxes.append([x1, y1, x2, y2])
        scores.append(p.score.value)
        class_ids.append(int(p.category.id))

    if len(boxes) == 0:
        return np.zeros((0,4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)

    return np.array(boxes, dtype=float), np.array(scores, dtype=float), np.array(class_ids, dtype=int)


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

@dataclass
class DetectionResult:
    boxes_xyxy: np.ndarray      # (N,4)
    class_ids: np.ndarray       # (N,)
    scores: np.ndarray          # (N,)
    track_ids: np.ndarray | None

    def filter(self, mask):
        return DetectionResult(
            boxes_xyxy=self.boxes_xyxy[mask],
            class_ids=self.class_ids[mask],
            scores=self.scores[mask],
            track_ids=None if self.track_ids is None else self.track_ids[mask],
        )
class UltralyticsDetector:
    def __init__(self, model, tracker_yaml, conf=0.1, iou=0.2):
        self.model = model
        self.tracker_yaml = tracker_yaml
        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            tracker=str(self.tracker_yaml),
            verbose=False,
        )
        r = results[0]

        if r.boxes.id is None:
            return DetectionResult(
                boxes_xyxy=np.zeros((0, 4)),
                class_ids=np.array([]),
                scores=np.array([]),
                track_ids=None,
            )

        return DetectionResult(
            boxes_xyxy=r.boxes.xyxy.cpu().numpy(),
            class_ids=r.boxes.cls.cpu().numpy().astype(int),
            scores=r.boxes.conf.cpu().numpy(),
            track_ids=r.boxes.id.cpu().numpy().astype(int),
        )

class SahiDetector:
    def __init__(self, sahi_model, run_every=5):
        self.model = sahi_model
        self.run_every = run_every
        self.last_result = None

    def detect(self, frame, frame_idx):
        if frame_idx % self.run_every != 0 and self.last_result is not None:
            return self.last_result

        boxes, scores, class_ids = sahi_detect_frame(
            frame,
            self.model,
        )

        self.last_result = DetectionResult(
            boxes_xyxy=boxes,
            class_ids=class_ids,
            scores=scores,
            track_ids=None,   # SAHI has no tracking
        )
        return self.last_result

import torch
def box_area_filter_idx(boxes, frame_shape, max_area_frac=0.04):
    """
    Returns boolean mask of boxes to keep.

    Supports:
    - Ultralytics Boxes object
    - (N,4) array-like boxes in xyxy format
    """
    H, W = frame_shape[:2]
    frame_area = H * W

    # Ultralytics Boxes
    if hasattr(boxes, "xyxy"):
        xyxy = boxes.xyxy
        if hasattr(xyxy, "cpu"):  # torch tensor
            xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.asarray(boxes)

    if xyxy.size == 0:
        return np.array([], dtype=bool)

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    area_frac = areas / frame_area

    return area_frac <= max_area_frac