"""Similarity fusion helpers for comparison scoring."""
from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_WEIGHTS = np.array([0.55, 0.15, 0.10, 0.10, 0.10], dtype=np.float32)
DEFAULT_THRESHOLD = 0.82


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Return a normalised weight vector."""

    arr = np.asarray(weights, dtype=np.float32)
    if arr.size != DEFAULT_WEIGHTS.size:
        raise ValueError("Expected %d weights" % DEFAULT_WEIGHTS.size)
    arr = np.clip(arr, 1e-4, None)
    total = float(arr.sum())
    if not total:
        return DEFAULT_WEIGHTS.copy()
    return arr / total


def compose_features(
    *,
    sim_embed: float,
    sim_color: float,
    sim_count: float,
    sim_size: float,
    sim_text: float = 0.0,
) -> np.ndarray:
    """Pack similarity components into a vector."""

    return np.array(
        [sim_embed, sim_color, sim_count, sim_size, sim_text],
        dtype=np.float32,
    )


def combine_scores(weights: Sequence[float], features: Sequence[float]) -> float:
    """Compute the fused score using the provided weights."""

    weight_vec = normalize_weights(weights)
    feature_vec = np.asarray(features, dtype=np.float32)
    if feature_vec.shape != weight_vec.shape:
        raise ValueError("Weights and features must share the same shape")
    score = float(np.dot(weight_vec, feature_vec))
    return float(np.clip(score, 0.0, 1.0))


def fuse_similarity(
    weights: Sequence[float],
    *,
    sim_embed: float,
    sim_color: float,
    sim_count: float,
    sim_size: float,
    sim_text: float = 0.0,
) -> float:
    """Convenience wrapper combining feature packing and score fusion."""

    features = compose_features(
        sim_embed=sim_embed,
        sim_color=sim_color,
        sim_count=sim_count,
        sim_size=sim_size,
        sim_text=sim_text,
    )
    return combine_scores(weights, features)

