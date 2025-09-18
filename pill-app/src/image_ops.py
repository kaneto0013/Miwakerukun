from __future__ import annotations

import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from . import detect, embed, schemas, store, visualize

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ImageProcessingError(Exception):
    """Base class for image processing related errors."""


class BagNotFoundError(ImageProcessingError):
    """Raised when a bag identifier cannot be resolved."""


class EmptyUploadError(ImageProcessingError):
    """Raised when an uploaded file contains no data."""


class InvalidImageError(ImageProcessingError):
    """Raised when the uploaded file is not a valid image."""


class ImageNotFoundError(ImageProcessingError):
    """Raised when an image identifier cannot be resolved."""


async def process_upload(bag_id: str, upload_file: UploadFile) -> schemas.Image:
    """Persist an uploaded image and run detection/embedding pipelines."""

    bag = store.get_bag_basic(bag_id)
    if bag is None:
        raise BagNotFoundError(bag_id)

    contents = await upload_file.read()
    if not contents:
        raise EmptyUploadError(upload_file.filename or "<unknown>")

    try:
        pil_image = Image.open(BytesIO(contents))
        pil_image.load()
        width, height = pil_image.size
    except (UnidentifiedImageError, OSError) as exc:  # pragma: no cover - depends on input data
        raise InvalidImageError(upload_file.filename or "<unknown>") from exc

    suffix = Path(upload_file.filename or "").suffix or ".png"
    filename = f"{uuid.uuid4()}{suffix}"
    file_path = RAW_DIR / filename
    with open(file_path, "wb") as output:
        output.write(contents)

    relative_path = str(file_path.relative_to(BASE_DIR))
    record = store.create_image(
        bag_id=bag_id,
        path=relative_path,
        width=width,
        height=height,
    )

    try:
        detections, image_size = detect.detect_pills(file_path)
    except Exception as exc:  # pragma: no cover - behaviour depends on optional deps
        logger.warning("Detection failed for %s: %s", record["id"], exc)
        detections = []
        image_size = (height, width)

    _persist_detections(record, pil_image, detections, image_size, file_path)
    pil_image.close()

    return schemas.Image(**record)


def reanalyze_image(image_id: str) -> schemas.ImageDetections:
    """Re-run detection and embedding for an existing image."""

    image = store.get_image(image_id)
    if image is None:
        raise ImageNotFoundError(image_id)

    image_path = BASE_DIR / image["path"]
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    try:
        pil_image = Image.open(image_path)
        pil_image.load()
    except (UnidentifiedImageError, OSError) as exc:  # pragma: no cover - depends on runtime files
        raise InvalidImageError(str(image_path)) from exc

    try:
        detections, image_size = detect.detect_pills(image_path)
    except Exception as exc:  # pragma: no cover - behaviour depends on optional deps
        logger.warning("Reanalysis failed for %s: %s", image_id, exc)
        detections = []
        image_size = (int(image["height"]), int(image["width"]))

    stored_detections, unrecognized, message, visualization_path = _persist_detections(
        image,
        pil_image,
        detections,
        image_size,
        image_path,
    )
    pil_image.close()

    return schemas.ImageDetections(
        image=schemas.Image(**image),
        detections=[schemas.Detection(**det) for det in stored_detections],
        visualization_path=visualization_path,
        unrecognized_regions=[schemas.RegionFlag(**region) for region in unrecognized],
        message=message,
    )


def load_image_with_detections(image_id: str) -> schemas.ImageDetections:
    """Return stored detections for an image, regenerating the overlay if needed."""

    image = store.get_image(image_id)
    if image is None:
        raise ImageNotFoundError(image_id)

    detections = store.list_detections(image_id)
    image_path = BASE_DIR / image["path"]
    image_size = (int(image["height"]), int(image["width"]))

    unrecognized, message = detect.identify_unrecognized_regions(detections, image_size)

    visualization_path = visualize.visualization_path_for(image_id, OUTPUT_DIR)
    if not visualization_path.exists() and image_path.exists():
        try:
            visualize.visualize_detections(image_path, detections, unrecognized, visualization_path)
        except FileNotFoundError:
            pass

    relative_visualization_path = str(visualization_path.relative_to(BASE_DIR))

    return schemas.ImageDetections(
        image=schemas.Image(**image),
        detections=[schemas.Detection(**det) for det in detections],
        visualization_path=relative_visualization_path,
        unrecognized_regions=[schemas.RegionFlag(**region) for region in unrecognized],
        message=message,
    )


def _persist_detections(
    image_record: Sequence[Tuple[str, object]] | dict[str, object],
    pil_image: Image.Image,
    detections: Iterable[dict[str, float]],
    image_size: Tuple[int, int],
    source_path: Path,
) -> Tuple[List[dict[str, object]], List[dict[str, object]], str | None, str]:
    """Store detections, embeddings and visualisation artefacts."""

    image_dict = dict(image_record)
    stored = store.replace_detections(image_dict["id"], detections)

    try:
        samples = []
        for det in stored:
            vector = embed.embed_crop(
                pil_image,
                (det["x1"], det["y1"], det["x2"], det["y2"]),
            )
            if vector.size != embed.EMBED_DIM:
                continue
            area = max(0, (det["x2"] - det["x1"]) * (det["y2"] - det["y1"]))
            samples.append(
                {
                    "id": str(uuid.uuid4()),
                    "detection_id": det["id"],
                    "embed": vector.astype("float32", copy=False).tobytes(),
                    "size_px": float(area),
                    "ocr_text": None,
                }
            )
        if samples:
            store.insert_samples(samples)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to compute embeddings for image %s", image_dict["id"])

    unrecognized, message = detect.identify_unrecognized_regions(detections, image_size)

    visualization_path = visualize.visualization_path_for(image_dict["id"], OUTPUT_DIR)
    try:
        visualize.visualize_detections(source_path, detections, unrecognized, visualization_path)
    except FileNotFoundError:
        pass

    relative_visualization_path = str(visualization_path.relative_to(BASE_DIR))

    return stored, list(unrecognized), message, relative_visualization_path

