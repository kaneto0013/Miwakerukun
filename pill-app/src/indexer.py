from __future__ import annotations

"""Utilities for computing similarity between embedding samples."""

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import]
except Exception:  # pragma: no cover - executed when faiss is absent
    faiss = None  # type: ignore[assignment]

from . import embed

logger = logging.getLogger(__name__)


def _extract_vectors(samples: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]]) -> np.ndarray:
    """Convert stored samples to a matrix of embeddings."""

    if not isinstance(samples, Sequence):
        samples = list(samples)
    vectors: list[np.ndarray] = []
    for sample in samples:
        raw = sample.get("embed")
        if raw is None:
            continue
        if isinstance(raw, memoryview):  # pragma: no branch - depends on sqlite version
            raw = raw.tobytes()
        if isinstance(raw, bytes):
            vector = np.frombuffer(raw, dtype=np.float32)
        else:
            vector = np.asarray(raw, dtype=np.float32)
        if vector.size != embed.EMBED_DIM:
            logger.debug("Skipping sample with unexpected dimensionality: %s", vector.size)
            continue
        vectors.append(np.array(vector, dtype=np.float32, copy=True))
    if not vectors:
        return np.empty((0, embed.EMBED_DIM), dtype=np.float32)
    return np.vstack(vectors)


def _mean_nearest_similarity(source: np.ndarray, target: np.ndarray) -> float:
    if source.size == 0 or target.size == 0:
        return 0.0
    source = np.ascontiguousarray(source, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)

    if faiss is not None:
        index = faiss.IndexFlatIP(embed.EMBED_DIM)
        index.add(target)
        distances, _ = index.search(source, 1)
        return float(distances.mean())

    similarities = source @ target.T
    if similarities.size == 0:
        return 0.0
    max_sim = similarities.max(axis=1)
    return float(np.mean(max_sim))


def mean_nearest_similarity(
    samples_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    samples_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
) -> float:
    """Compute the symmetric mean nearest-neighbour similarity between two bags."""

    vectors_a = _extract_vectors(samples_a)
    vectors_b = _extract_vectors(samples_b)

    score_ab = _mean_nearest_similarity(vectors_a, vectors_b)
    score_ba = _mean_nearest_similarity(vectors_b, vectors_a)

    if vectors_a.size == 0 and vectors_b.size == 0:
        return 0.0
    if vectors_a.size == 0:
        return score_ba
    if vectors_b.size == 0:
        return score_ab
    return float((score_ab + score_ba) / 2.0)
