"""brain.horizon_bandit — regime-conditioned horizon selection.

Thompson-style Beta posteriors per (canonical regime, horizon) cell,
updated from REAL trade closes only. The horizon classifier stays
mechanical; the bandit may only OVERRIDE when it has enough realized
data in this regime cell AND its preferred arm beats the classified
arm by BANDIT_MARGIN, plus a small exploration floor so every arm
keeps collecting data.

Horizon shapes patience, sizing bar, and exit windows — it never
touches the verdict itself (R1-safe strategy adaptation).
"""

from __future__ import annotations

import json
import logging
import random
import threading
from pathlib import Path
from typing import Any

from .learning_config import (
    BANDIT_EPS,
    BANDIT_FILE,
    BANDIT_MARGIN,
    BANDIT_MIN_SAMPLES,
    BANDIT_REWARD_SCALE,
)
from .meta_label import HORIZONS, REGIMES

log = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


class HorizonBandit:
    """Beta-bandit over horizons conditioned on the canonical regime."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or BANDIT_FILE
        self._lock = threading.RLock()
        self._arms: dict[str, dict[str, list[float]]] = {}
        self._overrides = 0
        self._explores = 0
        self._selects = 0
        self._load()

    def _cell(self, regime: str, horizon: str) -> list[float]:
        """Return the mutable [alpha, beta] cell (Beta(1, 1) prior)."""
        row = self._arms.setdefault(regime, {})
        return row.setdefault(horizon, [1.0, 1.0])

    @staticmethod
    def _mean(cell: list[float]) -> float:
        """Posterior mean of a Beta cell."""
        return cell[0] / (cell[0] + cell[1])

    @staticmethod
    def _trials(cell: list[float]) -> float:
        """Real trials in a cell (prior mass excluded)."""
        return max(0.0, cell[0] + cell[1] - 2.0)

    def select(self, regime: str, classified: str) -> tuple[str, str]:
        """Pick the horizon for this tick: classified, override, or explore.

        Returns (horizon, reason) — reason is one of "classifier",
        "bandit_override", "explore".
        """
        with self._lock:
            self._selects += 1
            if random.random() < BANDIT_EPS:
                return self._explore(regime, classified)
            best = self._best_arm(regime, classified)
            cls_cell = self._cell(regime, classified)
            if (
                best != classified
                and self._trials(cls_cell) >= BANDIT_MIN_SAMPLES
                and self._mean(self._cell(regime, best)) - self._mean(cls_cell)
                >= BANDIT_MARGIN
            ):
                self._overrides += 1
                return best, "bandit_override"
            return classified, "classifier"

    def _explore(self, regime: str, classified: str) -> tuple[str, str]:
        """Exploration draw: deviate only if the classified arm is trained.

        A cold cell (classified arm itself thin) keeps the classifier's
        choice — deviating with zero data is noise, not learning.
        """
        self._explores += 1
        if self._trials(self._cell(regime, classified)) < BANDIT_MIN_SAMPLES:
            return classified, "explore"
        thin = [
            h
            for h in HORIZONS
            if self._trials(self._cell(regime, h)) < BANDIT_MIN_SAMPLES
        ]
        if thin:
            pick = random.choice(thin)
            if pick != classified:
                return pick, "explore"
        return classified, "explore"

    def _best_arm(self, regime: str, classified: str) -> str:
        """Best-trained arm in this regime cell (classified as baseline)."""
        best, best_mean = classified, self._mean(self._cell(regime, classified))
        for h in HORIZONS:
            cell = self._cell(regime, h)
            if (
                self._trials(cell) >= BANDIT_MIN_SAMPLES
                and self._mean(cell) > best_mean
            ):
                best, best_mean = h, self._mean(cell)
        return best

    def update(self, regime: str, horizon: str, pnl_pct: float) -> None:
        """One REAL trade close → Beta update for its (regime, horizon) cell."""
        reward = _clamp01(0.5 + pnl_pct * BANDIT_REWARD_SCALE)
        with self._lock:
            cell = self._cell(regime, horizon)
            cell[0] += reward
            cell[1] += 1.0 - reward
            self._save()

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view: best arm per regime + selection counters."""
        with self._lock:
            per_regime: dict[str, Any] = {}
            for regime, row in self._arms.items():
                ranked = sorted(
                    ((h, self._mean(c), self._trials(c)) for h, c in row.items()),
                    key=lambda t: -t[1],
                )
                per_regime[regime] = [
                    {"horizon": h, "mean": round(m, 3), "n": int(n)}
                    for h, m, n in ranked
                ]
            return {
                "selects": self._selects,
                "overrides": self._overrides,
                "explores": self._explores,
                "arms": per_regime,
            }

    def _load(self) -> None:
        """Load persisted posteriors; corrupt/missing file → fresh priors."""
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            for regime, row in d.get("arms", {}).items():
                self._load_cell(regime, row)
            self._overrides = int(d.get("overrides", 0))
            self._explores = int(d.get("explores", 0))
            self._selects = int(d.get("selects", 0))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Horizon bandit load failed (fresh priors): %s", exc)

    def _load_cell(self, regime: str, row: dict[str, Any]) -> None:
        """Load one regime's arm cells from persisted posteriors."""
        if regime not in REGIMES:
            return
        for horizon, ab in row.items():
            if horizon in HORIZONS and isinstance(ab, list) and len(ab) == 2:
                self._cell(regime, horizon)
                self._arms[regime][horizon] = [
                    max(1.0, float(ab[0])),
                    max(1.0, float(ab[1])),
                ]

    def _save(self) -> None:
        """Persist atomically (tmp + replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "arms": self._arms,
                        "overrides": self._overrides,
                        "explores": self._explores,
                        "selects": self._selects,
                    }
                )
            )
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("Horizon bandit save failed: %s", exc)
