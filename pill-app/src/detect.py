"""YOLO-based pill detection utilities with contour fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO  # type: ignore[import]
except Exception as exc:  # pragma: no cover - executed when ultralytics is absent
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when ultralytics is available
    _YOLO_IMPORT_ERROR = None


@dataclass(frozen=True)
class Detection:
    """Representation of a detected object."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "score": float(self.score),
        }


@dataclass(frozen=True)
class RegionFlag:
    """Region flagged for manual review."""

    x1: int
    y1: int
    x2: int
    y2: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "reason": self.reason,
        }


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _prepare_image(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    height, width = image.shape[:2]
    return image, (height, width)


def _clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    height, width = image_size
    x1_i = int(np.clip(np.floor(x1), 0, max(0, width - 1)))
    y1_i = int(np.clip(np.floor(y1), 0, max(0, height - 1)))
    x2_i = int(np.clip(np.ceil(x2), x1_i + 1, width))
    y2_i = int(np.clip(np.ceil(y2), y1_i + 1, height))
    return x1_i, y1_i, x2_i, y2_i


@lru_cache(maxsize=1)
def _load_model(weights: str | None = None) -> object:
    """Lazily instantiate the Ultralytics YOLO model."""

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is unavailable") from _YOLO_IMPORT_ERROR
    weight_path = weights or "yolov8n.pt"
    try:
        return YOLO(weight_path)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - depends on runtime availability
        logger.warning("Failed to load YOLO model '%s': %s", weight_path, exc)
        raise


def _yolo_detect(
    image: np.ndarray,
    image_size: Tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> List[Detection]:
    model = _load_model()
    try:
        results = model.predict(  # type: ignore[attr-defined]
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime availability
        logger.warning("YOLO inference failed: %s", exc)
        return []

    height, width = image_size
    image_area = float(height * width)
    detections: list[Detection] = []

    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        confs = getattr(boxes, "conf", None)
        if xyxy is None or confs is None:
            continue
        coords = np.asarray(xyxy.cpu()) if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        scores = np.asarray(confs.cpu()) if hasattr(confs, "cpu") else np.asarray(confs)
        for coord, score in zip(coords, scores):
            confidence = float(score)
            if confidence <= 0.0:
                continue
            x1, y1, x2, y2 = (float(c) for c in coord[:4])
            x1_i, y1_i, x2_i, y2_i = _clip_box(x1, y1, x2, y2, image_size)
            w_box = max(0, x2_i - x1_i)
            h_box = max(0, y2_i - y1_i)
            if w_box < 2 or h_box < 2:
                continue
            area_ratio = float(w_box * h_box) / image_area if image_area else 0.0
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            detections.append(Detection(x1_i, y1_i, x2_i, y2_i, round(confidence, 3)))

    detections.sort(key=lambda det: det.score, reverse=True)
    return detections


def _contour_detect(
    image: np.ndarray,
    image_size: Tuple[int, int],
    blur_kernel_size: int,
    canny_thresholds: Tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    max_aspect_ratio: float,
) -> List[Detection]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_kernel = (blur_kernel_size, blur_kernel_size)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image_size
    image_area = float(height * width)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        area_ratio = area / image_area if image_area else 0.0
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
    return detections


def detect_pills(
    image_path: str | Path,
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.6,
    max_aspect_ratio: float = 6.0,
    blur_kernel_size: int = 5,
    canny_thresholds: Tuple[int, int] = (40, 120),
) -> Tuple[List[dict[str, float]], Tuple[int, int]]:
    """Detect pill-like regions in an image.

    YOLOv8 is used when available to provide robust bounding boxes. If the
    detector is unavailable or yields no detections the function falls back to a
    contour-based heuristic so existing workflows continue to function.
    """

    path = _to_path(image_path)
    image, image_size = _prepare_image(path)

    detections: list[Detection] = []
    if YOLO is not None:
        detections = _yolo_detect(
            image,
            image_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        if detections:
            logger.debug("YOLO produced %d detections", len(detections))
        else:
            logger.debug("YOLO produced no detections; falling back to contour method")
    else:
        logger.debug("Ultralytics not available (%s); using contour fallback", _YOLO_IMPORT_ERROR)

    if not detections:
        detections = _contour_detect(
            image,
            image_size,
            blur_kernel_size=blur_kernel_size,
            canny_thresholds=canny_thresholds,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )

    return [det.as_dict() for det in detections], image_size


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
    flags: list[RegionFlag] = []

    for det in detections:
        if det["score"] < low_score_threshold:
            flags.append(
                RegionFlag(
                    int(det["x1"]),
                    int(det["y1"]),
                    int(det["x2"]),
                    int(det["y2"]),
                    "low_score",
                )
            )

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

