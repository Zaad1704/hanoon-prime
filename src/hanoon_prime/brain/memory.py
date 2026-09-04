"""hanoon_prime.brain.memory — JULI's persistent learning memory.

Stores indicator weights, episodic patterns, score history, calibration
state, and structured lessons. Persists to JSON across restarts.

Thread-safe with RLock. Single-writer: reflection.py (on trade close).

Separate from memory.py (journal) — that's the IB carbon copy.
This is the BRAIN's learning state.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_WEIGHTS, EPISODIC_CAPACITY, JULI_STATE_FILE

log = logging.getLogger(__name__)

SAVE_KEYS = (
    "weights",
    "episodes",
    "scores",
    "all_scores",
    "pred_error_ema",
    "win_count",
    "loss_count",
    "lessons",
    "threshold",
)


class JuliMemory:
    """Persistent learning state for Juli's brain."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or JULI_STATE_FILE
        self._lock = threading.RLock()
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        self._episodes: list[dict[str, Any]] = []
        self._scores: dict[str, list[float]] = {}
        self._all_scores: list[float] = []
        self._pred_error_ema: float = 0.5
        self._win_count: int = 0
        self._loss_count: int = 0
        self._lessons: list[dict[str, Any]] = []
        self._threshold: float = 0.58
        self._load()

    def get_weights(self) -> dict[str, float]:
        """Return a copy of the current indicator weights."""
        with self._lock:
            return dict(self._weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Replace indicator weights and persist."""
        with self._lock:
            self._weights = dict(weights)
            self._save()

    def add_episode(
        self, vector: list[float], outcome: float, ticker: str = ""
    ) -> None:
        """Store one pattern episode with its outcome."""
        with self._lock:
            self._episodes.append(
                {
                    "vector": vector,
                    "outcome": outcome,
                    "ticker": ticker,
                    "ts": time.time(),
                }
            )
            if len(self._episodes) > EPISODIC_CAPACITY:
                self._episodes = self._episodes[-EPISODIC_CAPACITY:]
            self._save()

    def get_episodes(self) -> list[dict[str, Any]]:
        """Return a copy of all stored episodes."""
        with self._lock:
            return list(self._episodes)

    def record_score(self, ticker: str, score: float) -> None:
        """Record a score for percentile normalization."""
        with self._lock:
            self._scores.setdefault(ticker, []).append(score)
            self._all_scores.append(score)
            if len(self._all_scores) > 5000:
                self._all_scores = self._all_scores[-5000:]
            if len(self._scores[ticker]) > 500:
                self._scores[ticker] = self._scores[ticker][-500:]

    def get_score_history(self, ticker: str = "") -> list[float]:
        """Return score history for a ticker or all scores."""
        with self._lock:
            return list(self._scores.get(ticker, self._all_scores))

    def update_pred_error(self, predicted: float, actual: float) -> None:
        """Update the rolling prediction-error EMA."""
        error = abs(predicted - actual)
        with self._lock:
            self._pred_error_ema = 0.9 * self._pred_error_ema + 0.1 * error
            self._save()

    @property
    def pred_error(self) -> float:
        """Rolling prediction-error EMA."""
        return self._pred_error_ema

    def record_outcome(self, won: bool) -> None:
        """Record a trade win/loss."""
        with self._lock:
            if won:
                self._win_count += 1
            else:
                self._loss_count += 1
            self._save()

    @property
    def win_rate(self) -> float:
        """Rolling win rate over recorded trades."""
        total = self._win_count + self._loss_count
        return self._win_count / total if total > 0 else 0.5

    @property
    def total_trades(self) -> int:
        """Total number of recorded trade outcomes."""
        return self._win_count + self._loss_count

    def add_lesson(self, lesson: dict[str, Any]) -> None:
        """Store a structured lesson (capped, timestamped)."""
        with self._lock:
            lesson["_ts"] = time.time()
            self._lessons.append(lesson)
            if len(self._lessons) > 500:
                self._lessons = self._lessons[-250:]
            self._save()

    def get_lessons(self) -> list[dict[str, Any]]:
        """Return a copy of all stored lessons."""
        with self._lock:
            return list(self._lessons)

    @property
    def threshold(self) -> float:
        """Current entry threshold learned from prediction error."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the entry threshold, clamped to the valid range."""
        self._threshold = max(0.10, min(0.70, value))
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            for k in SAVE_KEYS:
                setattr(self, f"_{k}", d.get(k, getattr(self, f"_{k}")))
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("State load failed (starting fresh): %s", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: getattr(self, f"_{k}") for k in SAVE_KEYS}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, default=str))
        tmp.replace(self._path)

    def snapshot(self) -> dict[str, Any]:
        """Return a read-only snapshot for telemetry."""
        return {
            "weights": self.get_weights(),
            "episodes": len(self._episodes),
            "win_rate": round(self.win_rate, 3),
            "total_trades": self.total_trades,
            "pred_error_ema": round(self._pred_error_ema, 4),
            "threshold": self._threshold,
            "lessons": len(self._lessons),
        }
