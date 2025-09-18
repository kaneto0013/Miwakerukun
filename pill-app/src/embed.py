from __future__ import annotations

"""Embedding utilities for pill detections."""

import logging
from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
except Exception as exc:  # pragma: no cover - executed when torch is absent
    torch = None  # type: ignore[assignment]
    ResNet18_Weights = None  # type: ignore[assignment]
    resnet18 = None  # type: ignore[assignment]
    _BACKEND_ERROR = exc
else:  # pragma: no cover - exercised when torch is available
    _BACKEND_ERROR = None

EMBED_DIM = 512


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Convert various image representations to :class:`PIL.Image`."""

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            mode = "L"
        elif image.ndim == 3 and image.shape[2] == 1:
            mode = "L"
            image = image[:, :, 0]
        else:
            mode = "RGB"
        return Image.fromarray(image.astype(np.uint8), mode=mode)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


@lru_cache(maxsize=1)
def _load_model() -> Tuple[object | None, object | None]:
    """Load the ResNet18 backbone used for embeddings.

    The model is lazily instantiated to avoid the heavy import cost when the
    embedding endpoint is unused. If torch or torchvision are unavailable the
    function logs a warning and returns ``(None, None)``, causing callers to
    fall back to zero embeddings.
    """

    if torch is None or resnet18 is None or ResNet18_Weights is None:
        if _BACKEND_ERROR is not None:
            logger.warning("Embedding backend unavailable: %s", _BACKEND_ERROR)
        else:
            logger.warning("Embedding backend unavailable: torch/torchvision not installed")
        return None, None

    weights = ResNet18_Weights.IMAGENET1K_V1
    try:
        model = resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover - depends on runtime availability
        logger.warning("Falling back to randomly initialised ResNet18: %s", exc)
        model = resnet18(weights=None)
    model.fc = torch.nn.Identity()
    model.eval()
    for param in model.parameters():  # type: ignore[assignment]
        param.requires_grad_(False)
    device = torch.device("cpu")
    model.to(device)
    transform = weights.transforms()
    return model, transform


def _zero_vector() -> np.ndarray:
    return np.zeros(EMBED_DIM, dtype=np.float32)


def embed_crop(image: Image.Image | np.ndarray, box: Iterable[float | int]) -> np.ndarray:
    """Embed a cropped region of an image using ResNet18 features.

    Parameters
    ----------
    image:
        Source image either as a PIL image or a NumPy array.
    box:
        Bounding box ``(x1, y1, x2, y2)`` describing the crop.
    """

    pil_image = _as_pil(image).convert("RGB")
    width, height = pil_image.size
    x1, y1, x2, y2 = [int(float(coord)) for coord in box]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        logger.debug("Invalid crop bounds: (%s, %s, %s, %s)", x1, y1, x2, y2)
        return _zero_vector()

    crop = pil_image.crop((x1, y1, x2, y2))
    if crop.size[0] == 0 or crop.size[1] == 0:
        return _zero_vector()

    model, transform = _load_model()
    if model is None or transform is None:
        return _zero_vector()

    input_tensor = transform(crop).unsqueeze(0)
    with torch.inference_mode():  # type: ignore[operator]
        features = model(input_tensor)
    vector = features.detach().cpu().numpy().astype(np.float32)
    vector = vector.reshape(-1)
    if vector.size != EMBED_DIM:
        logger.debug("Unexpected embedding size: %s", vector.size)
        resized = np.zeros(EMBED_DIM, dtype=np.float32)
        resized[: min(EMBED_DIM, vector.size)] = vector[:EMBED_DIM]
        vector = resized
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32, copy=False)
