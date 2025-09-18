#!/usr/bin/env python3
"""Calibrate the decision threshold τ using recent feedback."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src import feedback, store  # noqa: E402

LOGGER = logging.getLogger("calibrate")


def _fetch_recent_feedback(limit: int) -> List[dict[str, object]]:
    """Return the latest feedback entries joined with comparison scores."""

    conn = store.get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT
                feedback.id,
                feedback.comparison_id,
                feedback.is_correct,
                feedback.operator,
                feedback.created_at,
                comparisons.s_total
            FROM feedback
            JOIN comparisons ON comparisons.id = feedback.comparison_id
            ORDER BY feedback.created_at DESC, feedback.id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    rows.reverse()  # chronological order (oldest -> newest)
    return rows


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_tau(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Calibrate τ so the sigmoid outputs match the empirical positive rate."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    if score_array.size == 0:
        raise ValueError("no samples available")
    positives = label_array.sum()
    if positives <= 0 or positives >= score_array.size:
        raise ValueError("both positive and negative feedback are required")

    target = positives / score_array.size
    lower = float(score_array.min()) - 10.0
    upper = float(score_array.max()) + 10.0
    for _ in range(80):
        mid = (lower + upper) / 2.0
        mean_prob = float(_sigmoid(score_array - mid).mean())
        if mean_prob >= target:
            lower = mid
        else:
            upper = mid
    calibrated = (lower + upper) / 2.0
    return float(np.clip(calibrated, 0.0, 1.0))


def _accuracy(scores: Sequence[float], labels: Sequence[int], tau: float) -> float:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int32)
    predictions = (score_array >= tau).astype(np.int32)
    return float((predictions == label_array).mean()) if score_array.size else float("nan")


def _summarise(records: Iterable[dict[str, object]]) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for record in records:
        score = float(record.get("s_total") or 0.0)
        label = 1 if int(record.get("is_correct") or 0) else 0
        scores.append(score)
        labels.append(label)
    return scores, labels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Number of recent feedback entries to use (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute τ without persisting the result",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    records = _fetch_recent_feedback(max(args.window, 1))
    if not records:
        LOGGER.error("No feedback entries found; aborting calibration")
        return 1

    scores, labels = _summarise(records)
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        LOGGER.error("Need both positive and negative feedback to calibrate τ")
        return 1

    params = feedback.get_state()
    current_tau = params.tau
    current_accuracy = _accuracy(scores, labels, current_tau)
    try:
        new_tau = _calibrate_tau(scores, labels)
    except ValueError as error:
        LOGGER.error("Calibration failed: %s", error)
        return 1
    calibrated_accuracy = _accuracy(scores, labels, new_tau)

    LOGGER.info("Evaluated %d feedback entries (positive rate %.1f%%)", len(scores), 100 * (sum(labels) / len(labels)))
    LOGGER.info("Current τ: %.4f (accuracy %.3f)", current_tau, current_accuracy)
    LOGGER.info("Calibrated τ: %.4f (accuracy %.3f)", new_tau, calibrated_accuracy)

    if args.dry_run:
        LOGGER.info("Dry-run mode enabled; τ not persisted")
        return 0

    feedback.set_tau(new_tau, persist=True)
    LOGGER.info("Persisted τ=%.4f to %s", new_tau, feedback.PARAMETER_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual script entry point
    raise SystemExit(main())
