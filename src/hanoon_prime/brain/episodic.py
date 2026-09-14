"""hanoon_prime.brain.episodic — pure numpy k-NN pattern memory.

Replaces Nash's XGBoost with cosine-distance nearest neighbors.
"Have I seen this pattern before? What happened?"

The modifier is BOUNDED (±EPISODIC_MOD_BOUND) — a dampener, not a gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    EPISODIC_CAPACITY,
    EPISODIC_K,
    EPISODIC_KEYS,
    EPISODIC_MIN_SAMPLES,
    EPISODIC_MOD_BOUND,
)


class EpisodicMemory:
    """k-NN episodic memory using cosine distance on normalized indicator vectors."""

    def __init__(self, capacity: int = EPISODIC_CAPACITY) -> None:
        self._capacity = capacity
        self._memory = np.zeros((capacity, len(EPISODIC_KEYS)), dtype=np.float32)
        self._outcomes = np.zeros(capacity, dtype=np.float32)
        self._contexts = ["" for _ in range(capacity)]
        self._size = 0
        self._pointer = 0

    def clear(self) -> None:
        """Reset memory to empty (ironclade cleanup)."""
        self._memory = np.zeros((self._capacity, len(EPISODIC_KEYS)), dtype=np.float32)
        self._outcomes = np.zeros(self._capacity, dtype=np.float32)
        self._contexts = ["" for _ in range(self._capacity)]
        self._size = 0
        self._pointer = 0

    def add(self, alpha: dict[str, float], outcome: float, context: str = "") -> None:
        """Store a pattern, its trade outcome, and the context it was learned in."""
        vec = self._build_vector(alpha)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._memory[self._pointer] = vec
        self._outcomes[self._pointer] = outcome
        self._contexts[self._pointer] = context
        self._pointer = (self._pointer + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def predict(
        self,
        alpha: dict[str, float],
        k: int = EPISODIC_K,
        context: str = "",
    ) -> tuple[float, float]:
        """Query: return (expected_return, confidence) for this situation.

        When ``context`` matches at least ``EPISODIC_MIN_SAMPLES`` tags, the
        neighbors are restricted to that context (context-gated recall);
        otherwise it falls back to the full buffer.
        """
        if self._size < EPISODIC_MIN_SAMPLES:
            return 0.0, 0.0
        query = self._build_vector(alpha)
        norm = np.linalg.norm(query)
        if norm == 0:
            return 0.0, 0.0
        query = query / norm
        active = self._memory[: self._size]
        outcomes = self._outcomes[: self._size]
        if context:
            sel = np.flatnonzero(np.array(self._contexts[: self._size]) == context)
            if len(sel) >= EPISODIC_MIN_SAMPLES:
                active = active[sel]
                outcomes = self._outcomes[sel]
        similarities = np.dot(active, query)
        k = min(k, len(active))
        top_idx = np.argpartition(similarities, -k)[-k:]
        top_sims = similarities[top_idx]
        top_outcomes = outcomes[top_idx]
        weights = np.exp(top_sims - np.max(top_sims))
        weights = weights / (np.sum(weights) + 1e-12)
        expected_return = float(np.sum(weights * top_outcomes))
        confidence = float(np.mean(top_sims))
        return expected_return, confidence

    def modifier(self, alpha: dict[str, float], context: str = "") -> float:
        """Return bounded modifier in [-BOUND, +BOUND], context-gated."""
        exp_ret, confidence = self.predict(alpha, context=context)
        if confidence < 0.3:
            return 0.0
        raw = exp_ret * confidence
        return float(max(-EPISODIC_MOD_BOUND, min(EPISODIC_MOD_BOUND, raw)))

    def recall_similar(
        self, alpha: dict[str, float], k: int = 5
    ) -> list[dict[str, Any]]:
        """Return k nearest episodes for reflection (human-readable)."""
        if self._size < EPISODIC_MIN_SAMPLES:
            return []
        query = self._build_vector(alpha)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm
        active = self._memory[: self._size]
        sims = np.dot(active, query)
        k = min(k, self._size)
        top_idx = np.argpartition(sims, -k)[-k:]
        return [
            {
                "won": bool(self._outcomes[i] > 0),
                "distance": float(1.0 - sims[i]),
                "outcome": float(self._outcomes[i]),
            }
            for i in top_idx
        ]

    @staticmethod
    def _build_vector(alpha: dict[str, float]) -> np.ndarray:
        """Map alpha dict to fixed-dimension feature vector."""
        return np.array(
            [float(alpha.get(k, 0.5)) for k in EPISODIC_KEYS], dtype=np.float32
        )

    @property
    def size(self) -> int:
        """Number of stored episodes."""
        return self._size
