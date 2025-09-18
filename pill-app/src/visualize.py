"""Visualization helpers for pill detection."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2


BLUE = (255, 0, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def visualization_path_for(image_id: str, output_dir: str | Path) -> Path:
    """Return the canonical visualization path for an image id."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{image_id}.png"


def visualize_detections(
    image_path: str | Path,
    detections: Sequence[dict[str, float]],
    unrecognized_regions: Sequence[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Draw detections and flagged regions and save the visualization image."""

    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    canvas = image.copy()

    for det in detections:
        pt1 = (int(det["x1"]), int(det["y1"]))
        pt2 = (int(det["x2"]), int(det["y2"]))
        cv2.rectangle(canvas, pt1, pt2, BLUE, 2)
        label = f"{det['score']:.2f}"
        text_origin = (pt1[0], max(15, pt1[1] - 5))
        cv2.putText(canvas, label, text_origin, FONT, 0.5, BLUE, 1, cv2.LINE_AA)

    for region in unrecognized_regions:
        pt1 = (int(region["x1"]), int(region["y1"]))
        pt2 = (int(region["x2"]), int(region["y2"]))
        cv2.rectangle(canvas, pt1, pt2, RED, 2)
        reason = str(region.get("reason", "check"))
        text_origin = (pt1[0], min(canvas.shape[0] - 5, pt2[1] + 15))
        cv2.putText(canvas, reason, text_origin, FONT, 0.5, RED, 1, cv2.LINE_AA)

    if not detections:
        cv2.putText(
            canvas,
            "再解析が必要です",
            (10, 30),
            FONT,
            0.8,
            RED,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path
