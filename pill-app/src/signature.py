"""Bag-of-Embeddings utilities for comparing pill embeddings."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import embed

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import MiniBatchKMeans
except Exception as exc:  # pragma: no cover - executed when scikit-learn is absent
    MiniBatchKMeans = None  # type: ignore[assignment]
    _SKLEARN_ERROR = exc
else:  # pragma: no cover - exercised when dependency is available
    _SKLEARN_ERROR = None


def _as_matrix(samples: Iterable[Mapping[str, object]] | np.ndarray) -> np.ndarray:
    """Convert stored samples to a 2D float32 array."""

    if isinstance(samples, np.ndarray):
        matrix = np.asarray(samples, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return np.ascontiguousarray(matrix, dtype=np.float32)

    vectors: list[np.ndarray] = []
    for sample in samples:
        raw = sample
        if isinstance(sample, Mapping):
            raw = sample.get("embed")
        if raw is None:
            continue
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, bytes):
            vector = np.frombuffer(raw, dtype=np.float32)
        else:
            vector = np.asarray(raw, dtype=np.float32)
        if vector.size != embed.EMBED_DIM:
            logger.debug(
                "Skipping embedding with unexpected dimensionality: %s",
                vector.size,
            )
            continue
        vectors.append(np.array(vector, dtype=np.float32, copy=True))

    if not vectors:
        return np.empty((0, embed.EMBED_DIM), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


def _chi_square_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    eps = 1e-8
    diff = hist_a - hist_b
    denom = hist_a + hist_b + eps
    return float(0.5 * np.sum((diff * diff) / denom))


def _cosine_similarity(vectors_a: np.ndarray, vectors_b: np.ndarray) -> float:
    if vectors_a.size == 0 or vectors_b.size == 0:
        return 0.0
    mean_a = vectors_a.mean(axis=0)
    mean_b = vectors_b.mean(axis=0)
    norm_a = float(np.linalg.norm(mean_a))
    norm_b = float(np.linalg.norm(mean_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cosine = float(np.dot(mean_a, mean_b) / (norm_a * norm_b))
    return float((cosine + 1.0) / 2.0)


@dataclass
class BagOfEmbeddings:
    """Maintain a shared MiniBatchKMeans model for embedding clustering."""

    n_clusters: int = 64
    batch_size: int = 256
    random_state: int | None = 0
    _model: MiniBatchKMeans | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _ensure_model(self) -> MiniBatchKMeans:
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn is unavailable") from _SKLEARN_ERROR
        if self._model is None:
            self._model = MiniBatchKMeans(  # type: ignore[call-arg]
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                n_init="auto",
                random_state=self.random_state,
            )
        return self._model

    def _prepare_training(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        if data.shape[0] >= self.n_clusters:
            return data
        reps = int(np.ceil(self.n_clusters / max(1, data.shape[0])))
        tiled = np.repeat(data, reps, axis=0)
        return tiled[: self.n_clusters]

    def fit_if_needed(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        model = self._ensure_model()
        training = self._prepare_training(data)
        try:
            model.partial_fit(training)
        except Exception as exc:  # pragma: no cover - depends on runtime data
            logger.warning("MiniBatchKMeans fitting failed: %s", exc)
            return
        self._fitted = True

    def histogram(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0 or not self._fitted or self._model is None:
            return np.zeros(self.n_clusters, dtype=np.float32)
        try:
            labels = self._model.predict(data)
        except Exception as exc:  # pragma: no cover - depends on runtime data
            logger.warning("Histogram prediction failed: %s", exc)
            return np.zeros(self.n_clusters, dtype=np.float32)
        hist = np.bincount(labels, minlength=self.n_clusters).astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def similarity(self, bag_a: np.ndarray, bag_b: np.ndarray) -> float:
        if bag_a.size == 0 and bag_b.size == 0:
            return 0.0
        combined: np.ndarray
        if bag_a.size and bag_b.size:
            combined = np.vstack([bag_a, bag_b])
        elif bag_a.size:
            combined = bag_a
        else:
            combined = bag_b

        self.fit_if_needed(combined)
        hist_a = self.histogram(bag_a)
        hist_b = self.histogram(bag_b)
        if hist_a.sum() == 0 and hist_b.sum() == 0:
            return 0.0
        distance = _chi_square_distance(hist_a, hist_b)
        similarity = float(1.0 / (1.0 + distance))
        return similarity


_MODEL = BagOfEmbeddings()


def boe_similarity(
    samples_a: Sequence[Mapping[str, object]] | np.ndarray,
    samples_b: Sequence[Mapping[str, object]] | np.ndarray,
) -> float:
    """Compute similarity between two bags using the BoE representation."""

    vectors_a = _as_matrix(samples_a)
    vectors_b = _as_matrix(samples_b)

    if MiniBatchKMeans is None:
        if _SKLEARN_ERROR is not None:
            logger.debug("BoE backend unavailable: %s", _SKLEARN_ERROR)
        return _cosine_similarity(vectors_a, vectors_b)

    try:
        return _MODEL.similarity(vectors_a, vectors_b)
    except RuntimeError:
        logger.debug("Falling back to cosine similarity due to missing backend")
        return _cosine_similarity(vectors_a, vectors_b)


def bag_histogram(samples: Sequence[Mapping[str, object]] | np.ndarray) -> np.ndarray:
    """Return the BoE histogram for a bag, fitting the model if necessary."""

    vectors = _as_matrix(samples)
    if MiniBatchKMeans is None:
        return np.zeros(_MODEL.n_clusters, dtype=np.float32)
    _MODEL.fit_if_needed(vectors)
    return _MODEL.histogram(vectors)

