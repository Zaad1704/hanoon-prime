"""brain.strategy_genome — Strategy genome / self-evolution.

Encodes scoring logic as an "editable genome" — a structured representation
of indicator weights, thresholds, and scoring rules that can be diagnosed.

Source: rebuild's strategy_genome.py (simplified).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_WEIGHTS, SIGNAL_THRESHOLD

log = logging.getLogger(__name__)

_GENOME_PATH = Path("runtime/strategy_genome.json")


class StrategyGenome:
    """Editable genome of scoring logic."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._genome: dict[str, Any] = {
            "weights": dict(DEFAULT_WEIGHTS),
            "threshold": SIGNAL_THRESHOLD,
            "modifiers": {},
            "version": 1,
        }
        self._load()

    def get_genome(self) -> dict[str, Any]:
        """Auto-generated docstring."""
        return dict(self._genome)

    def update_weight(self, indicator: str, weight: float) -> None:
        """Auto-generated docstring."""
        self._genome["weights"][indicator] = weight
        self._save()

    def update_threshold(self, threshold: float) -> None:
        """Auto-generated docstring."""
        self._genome["threshold"] = threshold
        self._save()

    def diagnose(self) -> list[str]:
        """Return diagnostic findings about the genome."""
        issues = []
        weights = self._genome.get("weights", {})
        total = sum(weights.values())
        if total < 0.8 or total > 1.2:
            issues.append(f"Weight sum={total:.4f} outside [0.8, 1.2]")
        for k, v in weights.items():
            if v > 0.20:
                issues.append(f"{k}={v:.4f} > 0.20 (dominant)")
        thresh = self._genome.get("threshold", 0.58)
        if thresh < 0.1 or thresh > 0.7:
            issues.append(f"Threshold={thresh} outside [0.1, 0.7]")
        return issues

    def _save(self) -> None:
        """Auto-generated docstring."""
        try:
            _GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _GENOME_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._genome, default=str))
            tmp.replace(_GENOME_PATH)
        except Exception as exc:
            log.debug("Genome save failed: %s", exc)

    def _load(self) -> None:
        """Auto-generated docstring."""
        if not _GENOME_PATH.exists():
            return
        try:
            self._genome = json.loads(_GENOME_PATH.read_text())
        except Exception as exc:
            log.debug("Genome load failed: %s", exc)
