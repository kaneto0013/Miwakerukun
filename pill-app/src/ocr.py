"""Utilities wrapping OCR engines for engraved pill markings."""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

from . import preprocess, store

logger = logging.getLogger(__name__)

AMBIGUOUS_RANGE: tuple[float, float] = (0.80, 0.86)
MAX_CANDIDATES = 6
_ALLOWED_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TRANSLATION = str.maketrans({"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})

try:  # pragma: no cover - optional dependency
    import pytesseract
except Exception as exc:  # pragma: no cover - optional dependency missing
    pytesseract = None  # type: ignore[assignment]
    _PYTESSERACT_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when pytesseract import works
    _PYTESSERACT_IMPORT_ERROR = None
    _TESSERACT_ERROR_CLASS = getattr(pytesseract, "TesseractError", RuntimeError)
    _TESSERACT_NOT_FOUND_CLASS = getattr(pytesseract, "TesseractNotFoundError", FileNotFoundError)

try:  # pragma: no cover - optional dependency
    import easyocr
except Exception as exc:  # pragma: no cover - optional dependency missing
    easyocr = None  # type: ignore[assignment]
    _EASYOCR_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when easyocr import works
    _EASYOCR_IMPORT_ERROR = None

_TESSERACT_USABLE: bool | None = None


@dataclass
class SampleCandidate:
    sample_id: str
    bbox: tuple[int, int, int, int]
    image_path: Path
    ocr_text: str | None
    ratio: float
    area: int


def is_score_ambiguous(score: float, *, lower: float | None = None, upper: float | None = None) -> bool:
    """Return ``True`` when ``score`` lies within the configured ambiguous band."""

    lo = AMBIGUOUS_RANGE[0] if lower is None else float(lower)
    hi = AMBIGUOUS_RANGE[1] if upper is None else float(upper)
    if lo > hi:
        lo, hi = hi, lo
    return lo <= float(score) <= hi


def score_samples(
    samples_a: Sequence[dict[str, object]],
    samples_b: Sequence[dict[str, object]],
    base_dir: Path | str,
    *,
    max_candidates: int = MAX_CANDIDATES,
    ngram_size: int = 2,
) -> float:
    """Compute the OCR similarity score for two sample collections."""

    base_path = Path(base_dir)
    texts_a = _collect_texts(samples_a, base_path, max_candidates=max_candidates)
    texts_b = _collect_texts(samples_b, base_path, max_candidates=max_candidates)
    if not texts_a or not texts_b:
        logger.debug(
            "OCR skipped due to insufficient text candidates (A=%d, B=%d)",
            len(texts_a),
            len(texts_b),
        )
        return 0.0
    score = ngram_similarity(texts_a, texts_b, n=ngram_size)
    logger.info(
        "OCR similarity=%.3f (ngrams=%d, texts_a=%s, texts_b=%s)",
        score,
        ngram_size,
        texts_a,
        texts_b,
    )
    return score


def ngram_similarity(texts_a: Iterable[str], texts_b: Iterable[str], *, n: int = 2) -> float:
    """Return the best Jaccard similarity between ``n``-gram sets of both corpora."""

    sets_a = [_text_to_ngrams(text, n) for text in texts_a]
    sets_b = [_text_to_ngrams(text, n) for text in texts_b]
    sets_a = [s for s in sets_a if s]
    sets_b = [s for s in sets_b if s]
    if not sets_a or not sets_b:
        return 0.0

    best = 0.0
    for grams_a in sets_a:
        for grams_b in sets_b:
            union = grams_a | grams_b
            if not union:
                continue
            intersection = grams_a & grams_b
            if not intersection:
                continue
            score = len(intersection) / len(union)
            best = max(best, score)
            if best >= 1.0:  # pragma: no branch - early exit on perfect match
                return 1.0
    return float(best)


def recognize_text(
    image: np.ndarray | Image.Image,
    *,
    languages: Sequence[str] = ("eng",),
    allowlist: str = _ALLOWED_CHARACTERS,
) -> str:
    """Run OCR using the available backend and return the recognised text."""

    pil_image = _ensure_pil(image)
    text = _run_pytesseract(pil_image, languages, allowlist)
    if text:
        return text

    if easyocr is not None:
        text = _run_easyocr(pil_image, languages)
        if text:
            return text

    return ""


def _collect_texts(
    samples: Sequence[dict[str, object]],
    base_path: Path,
    *,
    max_candidates: int,
) -> List[str]:
    candidates = _select_candidates(samples, max_candidates)
    if not candidates:
        return []

    results: List[str] = []
    cache: dict[Path, Image.Image] = {}
    try:
        for candidate in candidates:
            cached_text = candidate.ocr_text
            if cached_text is not None:
                normalised = _normalise_text(cached_text)
                if normalised:
                    results.append(normalised)
                continue

            image_path = base_path / candidate.image_path
            image = cache.get(image_path)
            if image is None:
                try:
                    image = Image.open(image_path)
                    image.load()
                except (FileNotFoundError, OSError) as exc:
                    logger.warning("Failed to open %s for OCR: %s", image_path, exc)
                    store.update_sample_ocr(candidate.sample_id, "")
                    continue
                cache[image_path] = image

            crop = image.crop(candidate.bbox)
            cleaned = _perform_ocr(crop)
            store.update_sample_ocr(candidate.sample_id, cleaned)
            if cleaned:
                results.append(cleaned)
    finally:
        for image in cache.values():
            image.close()
    return results


def _select_candidates(
    samples: Sequence[dict[str, object]],
    max_candidates: int,
) -> List[SampleCandidate]:
    candidates: List[SampleCandidate] = []
    for sample in samples:
        path = sample.get("image_path")
        if not path:
            continue
        try:
            bbox = (
                int(sample["x1"]),
                int(sample["y1"]),
                int(sample["x2"]),
                int(sample["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        width = int(sample.get("image_width") or 0)
        height = int(sample.get("image_height") or 0)
        area = max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area <= 0 or width <= 0 or height <= 0:
            continue
        ratio = area / float(width * height)
        if ratio < 0.002 or ratio > 0.35:
            continue
        text = sample.get("ocr_text")
        if text == "":
            cached = ""
        else:
            cached = text if isinstance(text, str) else None
        candidates.append(
            SampleCandidate(
                sample_id=str(sample["id"]),
                bbox=bbox,
                image_path=Path(str(path)),
                ocr_text=cached,
                ratio=ratio,
                area=area,
            )
        )

    candidates.sort(
        key=lambda c: (
            0
            if c.ocr_text and _normalise_text(c.ocr_text)
            else 1
            if c.ocr_text is None
            else 2,
            c.ratio,
            -c.area,
        )
    )
    return candidates[:max_candidates]


def _perform_ocr(image: Image.Image) -> str:
    """Return normalised OCR text for a cropped image region."""

    processed = preprocess.enhance_contrast(image, top_hat_size=11, black_hat_size=31)
    binary = preprocess.adaptive_threshold(processed, block_size=25, c=7, invert=True)
    for candidate in (binary, processed, preprocess.to_grayscale(image)):
        text = recognize_text(candidate)
        normalised = _normalise_text(text)
        if normalised:
            return normalised
    return ""


def _normalise_text(text: str | None) -> str:
    if not text:
        return ""
    normalised = unicodedata.normalize("NFKC", str(text)).upper()
    normalised = normalised.translate(_TRANSLATION)
    normalised = re.sub(r"[^A-Z0-9]", "", normalised)
    return normalised


def _text_to_ngrams(text: str, n: int) -> set[str]:
    cleaned = _normalise_text(text)
    if not cleaned:
        return set()
    if len(cleaned) <= n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def _ensure_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L")
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
            return Image.fromarray(array.astype(np.uint8), mode="L")
        return Image.fromarray(array.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _run_pytesseract(
    image: Image.Image,
    languages: Sequence[str],
    allowlist: str,
) -> str:
    global _TESSERACT_USABLE
    if pytesseract is None:
        if _PYTESSERACT_IMPORT_ERROR is not None:
            logger.debug("pytesseract unavailable: %s", _PYTESSERACT_IMPORT_ERROR)
        return ""
    if _TESSERACT_USABLE is False:
        return ""

    lang = "+".join(languages) if languages else "eng"
    config = "--psm 6"
    if allowlist:
        config += f" -c tessedit_char_whitelist={allowlist}"
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except _TESSERACT_NOT_FOUND_CLASS as exc:  # type: ignore[func-returns-value]
        logger.warning("Tesseract binary not available: %s", exc)
        _TESSERACT_USABLE = False
        return ""
    except _TESSERACT_ERROR_CLASS as exc:  # type: ignore[func-returns-value]
        logger.debug("Tesseract OCR failed: %s", exc)
        _TESSERACT_USABLE = False
        return ""
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Unexpected Tesseract failure: %s", exc)
        return ""
    _TESSERACT_USABLE = True
    return text.strip()


@lru_cache(maxsize=1)
def _load_easyocr_reader(languages: tuple[str, ...]) -> object | None:
    if easyocr is None:
        if _EASYOCR_IMPORT_ERROR is not None:
            logger.debug("easyocr unavailable: %s", _EASYOCR_IMPORT_ERROR)
        return None
    try:
        return easyocr.Reader(list(languages), gpu=False, verbose=False)
    except Exception as exc:  # pragma: no cover - depends on runtime availability
        logger.warning("EasyOCR unavailable: %s", exc)
        return None


def _run_easyocr(image: Image.Image, languages: Sequence[str]) -> str:
    reader = _load_easyocr_reader(tuple(languages))
    if reader is None:
        return ""
    array = np.array(image.convert("RGB"))
    try:
        results = reader.readtext(array, detail=0, paragraph=True)
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.debug("EasyOCR failed: %s", exc)
        return ""
    text = " ".join(result.strip() for result in results if result).strip()
    return text
