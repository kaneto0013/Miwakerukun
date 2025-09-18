"""Adaptive parameter management for comparison scoring."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable


logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParameters:
    """Track and adjust scoring parameters using an EMA update."""

    w: float = 1.0
    tau: float = 0.1
    decay: float = 0.9
    learning_rate: float = 0.05
    _grad_w: float = field(default=0.0, init=False, repr=False)
    _grad_tau: float = field(default=0.0, init=False, repr=False)

    def compute_total(self, partial_scores: Iterable[float]) -> float:
        """Compute a total score using the current adaptive parameters."""

        scores = list(float(score) for score in partial_scores)
        if not scores:
            base = 0.0
        else:
            base = sum(scores) / len(scores)
        total = self.w * base + self.tau
        logger.info(
            "Computed s_total=%.6f (base=%.6f, w=%.4f, tau=%.4f)",
            total,
            base,
            self.w,
            self.tau,
        )
        return total

    def register_feedback(self, is_correct: bool) -> None:
        """Update the parameters based on user feedback.

        When feedback is correct the update nudges the parameters to become
        smaller, effectively lowering the total score. Incorrect feedback
        increases the parameters. Exponential moving averages are used to make
        consecutive feedback signals more influential, emulating a light-weight
        AdaDelta-like behaviour.
        """

        direction = -1.0 if is_correct else 1.0
        self._grad_w = self.decay * self._grad_w + (1.0 - self.decay) * direction
        self._grad_tau = self.decay * self._grad_tau + (1.0 - self.decay) * direction
        delta_w = self.learning_rate * self._grad_w
        delta_tau = self.learning_rate * self._grad_tau

        old_w, old_tau = self.w, self.tau

        self.w = max(0.05, self.w + delta_w)
        self.tau = max(0.0, self.tau + delta_tau)

        logger.info(
            "Feedback %s -> grad_w=%.6f grad_tau=%.6f; w: %.4f -> %.4f, tau: %.4f -> %.4f",
            "correct" if is_correct else "incorrect",
            self._grad_w,
            self._grad_tau,
            old_w,
            self.w,
            old_tau,
            self.tau,
        )


_STATE = AdaptiveParameters()


def get_state() -> AdaptiveParameters:
    """Return the singleton adaptive parameter state."""

    return _STATE


def reset_state() -> None:
    """Reset parameters to their defaults (mainly for tests)."""

    global _STATE
    _STATE = AdaptiveParameters()
