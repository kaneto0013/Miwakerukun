"""Utility functions for basic pill detection using OpenCV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Representation of a detected object."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "score": float(self.score)}


@dataclass(frozen=True)
class RegionFlag:
    """Region flagged for manual review."""

    x1: int
    y1: int
    x2: int
    y2: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "reason": self.reason}


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _prepare_image(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return gray, (height, width)


def detect_pills(
    image_path: str | Path,
    blur_kernel_size: int = 5,
    canny_thresholds: Tuple[int, int] = (40, 120),
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.6,
    max_aspect_ratio: float = 6.0,
) -> Tuple[List[dict[str, float]], Tuple[int, int]]:
    """Detect pill-like contours in an image."""

    path = _to_path(image_path)
    gray, (height, width) = _prepare_image(path)

    blur_kernel = (blur_kernel_size, blur_kernel_size)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(height * width)
    detections: List[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        area_ratio = area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        x, y, w_box, h_box = cv2.boundingRect(contour)
        if w_box < 5 or h_box < 5:
            continue

        aspect_ratio = max(w_box, h_box) / max(1, min(w_box, h_box))
        if aspect_ratio > max_aspect_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = float((4 * np.pi * area) / (perimeter * perimeter))
        box_area = float(w_box * h_box)
        fill_ratio = area / box_area if box_area else 0.0

        norm_area = min(1.0, area_ratio / 0.05)
        fill_score = min(1.0, fill_ratio)
        aspect_score = max(0.0, 1.0 - (aspect_ratio - 1.0) / 4.0)
        circularity_score = min(1.0, circularity)
        raw_score = 0.35 * fill_score + 0.25 * aspect_score + 0.2 * norm_area + 0.2 * circularity_score
        score = max(0.0, min(1.0, raw_score))

        detections.append(Detection(x, y, x + w_box, y + h_box, round(score, 3)))

    detections.sort(key=lambda det: det.score, reverse=True)
    return [det.as_dict() for det in detections], (height, width)


def _intersection_over_union(a: Detection | dict[str, float], b: Detection | dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = int(a["x1"]), int(a["y1"]), int(a["x2"]), int(a["y2"])
    bx1, by1, bx2, by2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def identify_unrecognized_regions(
    detections: Sequence[dict[str, float]],
    image_size: Tuple[int, int],
    low_score_threshold: float = 0.45,
    cluster_iou_threshold: float = 0.4,
) -> Tuple[List[dict[str, object]], str | None]:
    """Identify regions that require manual verification."""

    height, width = image_size
    flags: List[RegionFlag] = []

    for det in detections:
        if det["score"] < low_score_threshold:
            flags.append(RegionFlag(int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"]), "low_score"))

    clustered: set[Tuple[int, int, int, int]] = set()
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det_a, det_b = detections[i], detections[j]
            iou = _intersection_over_union(det_a, det_b)
            if iou >= cluster_iou_threshold:
                x1 = min(int(det_a["x1"]), int(det_b["x1"]))
                y1 = min(int(det_a["y1"]), int(det_b["y1"]))
                x2 = max(int(det_a["x2"]), int(det_b["x2"]))
                y2 = max(int(det_a["y2"]), int(det_b["y2"]))
                clustered.add((x1, y1, x2, y2))

    for x1, y1, x2, y2 in clustered:
        flags.append(RegionFlag(x1, y1, x2, y2, "clustered"))

    message: str | None = None
    if not detections:
        flags.append(RegionFlag(0, 0, width, height, "no_detections"))
        message = "検出に失敗しました。再解析が必要です。"
    elif flags:
        message = "一部の領域で再解析を推奨します。"

    return [flag.as_dict() for flag in flags], message
