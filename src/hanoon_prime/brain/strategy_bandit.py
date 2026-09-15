"""brain.strategy_bandit — regime-conditioned strategy selection.

Thompson Beta bandit conditioned on canonical regime; decaying ε and
lock-in gates keep exploration generous early and disciplined late.
Posteriors update from IRONCLADE-gated real closes only.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
from pathlib import Path
from typing import Any

from .arm_stats import beta_variance
from .learning_config import (
    STRATEGY_BANDIT_FILE,
    STRATEGY_EPS0,
    STRATEGY_EPS_DECAY,
    STRATEGY_EPS_FLOOR,
    STRATEGY_MARGIN,
    STRATEGY_MIN_SAMPLES,
    STRATEGY_REWARD_SCALE,
)
from .meta_label import REGIMES

log = logging.getLogger(__name__)
DEFAULT_STRATEGY: str = "default"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class StrategyBandit:
    """Beta-bandit over strategy ids conditioned on the canonical regime."""

    def __init__(self, path: Path | None = None) -> None:
        """Load (or start fresh) posteriors and the decay clock."""
        self._path = path or STRATEGY_BANDIT_FILE
        self._lock = threading.RLock()
        self._cells: dict[str, dict[str, list[float]]] = {}
        self._selects: int = 0
        self._overrides: int = 0
        self._explores: int = 0
        self._total_trials: int = 0
        self._load()

    def _cell(self, regime: str, strategy_id: str) -> list[float]:
        """Mutable [alpha, beta] cell (Beta(1, 1) prior)."""
        row = self._cells.setdefault(regime, {})
        return row.setdefault(strategy_id, [1.0, 1.0])

    @staticmethod
    def _mean(cell: list[float]) -> float:
        """Posterior mean of a Beta cell."""
        return cell[0] / (cell[0] + cell[1])

    @staticmethod
    def _trials(cell: list[float]) -> float:
        """Real trials in a cell (prior mass excluded)."""
        return max(0.0, cell[0] + cell[1] - 2.0)

    def _eps(self) -> float:
        """Decaying exploration: generous early, floor late."""
        decay = max(1.0, self._total_trials / max(1.0, STRATEGY_EPS_DECAY))
        return max(STRATEGY_EPS_FLOOR, STRATEGY_EPS0 / math.sqrt(decay))

    def select(self, regime: str, ids: list[str]) -> tuple[str, str]:
        """Pick strategy for this tick (lock-in, explore, or default)."""
        with self._lock:
            self._selects += 1
            if random.random() < self._eps():
                return self._explore(regime, ids)
            best = self._best_arm(regime, ids)
            def_cell = self._cell(regime, DEFAULT_STRATEGY)
            if (
                best != DEFAULT_STRATEGY
                and self._trials(def_cell) >= STRATEGY_MIN_SAMPLES
                and self._mean(self._cell(regime, best)) - self._mean(def_cell)
                >= STRATEGY_MARGIN
            ):
                self._overrides += 1
                return best, "strategy_override"
            return DEFAULT_STRATEGY, "default"

    def _explore(self, regime: str, ids: list[str]) -> tuple[str, str]:
        """Exploration draw: thin strategies only; else default."""
        self._explores += 1
        thin = [
            sid
            for sid in ids
            if self._trials(self._cell(regime, sid)) < STRATEGY_MIN_SAMPLES
        ]
        if thin and regime in self._cells:
            pick = random.choice(thin)
            empty = self._cell(regime, DEFAULT_STRATEGY)
            if empty and self._trials(empty) >= STRATEGY_MIN_SAMPLES:
                return pick, "explore"
        return DEFAULT_STRATEGY, "default"

    def _best_arm(self, regime: str, ids: list[str]) -> str:
        """Best-trained strategy in this regime (default is the baseline)."""
        best, best_mean = DEFAULT_STRATEGY, self._mean(
            self._cell(regime, DEFAULT_STRATEGY)
        )
        for sid in ids:
            cell = self._cell(regime, sid)
            if (
                self._trials(cell) >= STRATEGY_MIN_SAMPLES
                and self._mean(cell) > best_mean
            ):
                best, best_mean = sid, self._mean(cell)
        return best

    def update(self, regime: str, strategy_id: str, pnl_pct: float) -> None:
        """One real close → Beta update for (regime, strategy) cell."""
        reward = _clamp01(0.5 + pnl_pct * STRATEGY_REWARD_SCALE)
        with self._lock:
            cell = self._cell(regime, strategy_id)
            cell[0] += reward
            cell[1] += 1.0 - reward
            self._total_trials += 1
            self._save()

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view: per-regime strategy ranking + counters."""
        with self._lock:
            per_regime: dict[str, Any] = {}
            for regime, row in self._cells.items():
                per_regime[regime] = [
                    {
                        "strategy": sid,
                        "mean": round(self._mean(c), 3),
                        "n": int(self._trials(c)),
                        "ab": [round(c[0], 3), round(c[1], 3)],
                        "variance": round(beta_variance(c[0], c[1]), 4),
                    }
                    for sid, c in sorted(row.items(), key=lambda kv: -self._mean(kv[1]))
                ]
            return {
                "selects": self._selects,
                "overrides": self._overrides,
                "explores": self._explores,
                "eps": round(self._eps(), 3),
                "total_trials": self._total_trials,
                "arms": per_regime,
            }

    def reset(self) -> None:
        """Wipe posteriors and the decay clock."""
        with self._lock:
            self._cells = {}
            self._selects = self._overrides = self._explores = self._total_trials = 0
            self._save()

    def _load(self) -> None:
        """Load persisted posteriors; corrupt/missing file → fresh."""
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            for regime, row in d.get("cells", {}).items():
                if regime not in REGIMES:
                    continue
                self._cells[regime] = {
                    sid: [max(1.0, float(ab[0])), max(1.0, float(ab[1]))]
                    for sid, ab in row.items()
                    if isinstance(ab, list) and len(ab) == 2
                }
            self._selects = int(d.get("selects", 0))
            self._overrides = int(d.get("overrides", 0))
            self._explores = int(d.get("explores", 0))
            self._total_trials = int(d.get("total_trials", 0))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Strategy bandit load failed (fresh): %s", exc)

    def _save(self) -> None:
        """Persist atomically (tmp + replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "cells": self._cells,
                        "selects": self._selects,
                        "overrides": self._overrides,
                        "explores": self._explores,
                        "total_trials": self._total_trials,
                    }
                )
            )
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("Strategy bandit save failed: %s", exc)
