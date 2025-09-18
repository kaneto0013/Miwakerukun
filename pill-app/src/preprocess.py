"""Pre-processing utilities for stamp/engraving OCR."""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency in some environments
    from PIL import Image
except Exception:  # pragma: no cover - handled when Pillow missing
    Image = None  # type: ignore

__all__ = [
    "to_grayscale",
    "top_hat",
    "black_hat",
    "enhance_contrast",
    "adaptive_threshold",
]


ArrayLike = Iterable[Iterable[int]] | np.ndarray


def _as_array(image: ArrayLike | "Image.Image") -> np.ndarray:
    """Convert input image representations to a ``uint8`` NumPy array."""

    if Image is not None and isinstance(image, Image.Image):
        return np.array(image.convert("L"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim != 2:
        raise TypeError(f"Unsupported image shape: {arr.shape!r}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_grayscale(image: ArrayLike | "Image.Image") -> np.ndarray:
    """Return a grayscale ``uint8`` image suitable for morphological ops."""

    return _as_array(image)


def _kernel(size: int | Tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    size = tuple(max(1, int(s)) for s in size)
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)


def top_hat(image: ArrayLike | "Image.Image", size: int | Tuple[int, int] = 15) -> np.ndarray:
    """Enhance bright ridges on a dark background using the top-hat transform."""

    gray = to_grayscale(image)
    kernel = _kernel(size)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def black_hat(image: ArrayLike | "Image.Image", size: int | Tuple[int, int] = 31) -> np.ndarray:
    """Enhance dark grooves on a bright background using the black-hat transform."""

    gray = to_grayscale(image)
    kernel = _kernel(size)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def enhance_contrast(
    image: ArrayLike | "Image.Image",
    *,
    top_hat_size: int | Tuple[int, int] = 15,
    black_hat_size: int | Tuple[int, int] = 31,
) -> np.ndarray:
    """Combine top-hat/black-hat to emphasise engraved characters."""

    gray = to_grayscale(image)
    bright = top_hat(gray, top_hat_size)
    dark = black_hat(gray, black_hat_size)
    enhanced = cv2.add(gray, bright)
    enhanced = cv2.subtract(enhanced, dark)
    return enhanced


def adaptive_threshold(
    image: ArrayLike | "Image.Image",
    *,
    block_size: int = 35,
    c: int = 10,
    invert: bool = True,
    method: str = "gaussian",
) -> np.ndarray:
    """Return an adaptive binary mask of the image."""

    gray = to_grayscale(image)
    block = max(3, int(block_size))
    if block % 2 == 0:
        block += 1
    method_name = method.lower()
    method_flag = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method_name.startswith("g")
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, method_flag, thresh_type, block, int(c))
