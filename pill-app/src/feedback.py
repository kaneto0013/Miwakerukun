"""Adaptive parameter management for comparison scoring."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from . import match

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParameters:
    """Track scoring weights and adjust them using an EMA update."""

    weights: list[float] = field(default_factory=lambda: match.DEFAULT_WEIGHTS.tolist())
    tau: float = match.DEFAULT_THRESHOLD
    decay: float = 0.9
    learning_rate: float = 0.05
    _grad_tau: float = field(default=0.0, init=False, repr=False)

    def compute_total(
        self,
        *,
        sim_embed: float,
        sim_color: float,
        sim_count: float,
        sim_size: float,
        sim_text: float = 0.0,
    ) -> float:
        """Compute a fused similarity score using the current weights."""

        features = match.compose_features(
            sim_embed=sim_embed,
            sim_color=sim_color,
            sim_count=sim_count,
            sim_size=sim_size,
            sim_text=sim_text,
        )
        total = match.combine_scores(self.weights, features)
        offset = (self.tau - match.DEFAULT_THRESHOLD) * 0.25
        adjusted = float(np.clip(total + offset, 0.0, 1.0))
        logger.info(
            "Computed s_total=%.6f (base=%.6f, embed=%.4f, color=%.4f, count=%.4f, size=%.4f, text=%.4f, tau=%.4f)",
            adjusted,
            total,
            sim_embed,
            sim_color,
            sim_count,
            sim_size,
            sim_text,
            self.tau,
        )
        return adjusted

    def register_feedback(self, is_correct: bool) -> None:
        """Update weights in response to feedback."""

        direction = -1.0 if is_correct else 1.0
        old_weights = self.weights.copy()
        current = np.asarray(self.weights, dtype=np.float32)
        primary = float(
            np.clip(current[0] - direction * self.learning_rate, 0.05, 0.85)
        )
        remainder = max(1e-3, 1.0 - primary)
        tail = current[1:]
        if tail.sum() <= 0.0:
            tail = np.ones_like(tail) / max(1, tail.size)
        tail = tail / tail.sum()
        tail = np.clip(tail, 1e-3, 1.0)
        tail = tail / tail.sum()
        tail *= remainder
        updated = np.concatenate(([primary], tail))
        updated = match.normalize_weights(updated)
        self.weights = updated.tolist()

        self._grad_tau = self.decay * self._grad_tau + (1.0 - self.decay) * direction
        old_tau = self.tau
        self.tau = float(np.clip(self.tau + self.learning_rate * self._grad_tau, 0.0, 1.0))

        logger.info(
            "Feedback %s -> weights: %s -> %s, tau: %.4f -> %.4f",
            "correct" if is_correct else "incorrect",
            [f"{w:.3f}" for w in old_weights],
            [f"{w:.3f}" for w in self.weights],
            old_tau,
            self.tau,
        )

    @property
    def w(self) -> float:
        """Return the primary (embedding) weight for backwards compatibility."""

        return float(self.weights[0])


_STATE = AdaptiveParameters()


def get_state() -> AdaptiveParameters:
    """Return the singleton adaptive parameter state."""

    return _STATE


def reset_state() -> None:
    """Reset parameters to their defaults (mainly for tests)."""

    global _STATE
    _STATE = AdaptiveParameters()

