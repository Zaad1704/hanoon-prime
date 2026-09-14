"""hanoon_prime.brain.metacog — MetaMonitor: confidence-of-confidence.

Second-order tracking: how well does our stated confidence forecast
actual outcomes? A rolling correlation over (confidence bin, outcome)
pairs. When the calibration degrades, sizing shrinks — never bet big
on a brain that cannot price its own uncertainty.

Surprise (a situation matching no known pattern) plus pillar health
feeds a curiosity drive: explore when everything else is stable,
retreat when the pillar is falling. Advisory by construction — it
shapes sizing only, never a verdict (R1). Persists to
``runtime/juli_metacog.json`` (``HANOO_METACOG_FILE`` override).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

from .learning_config import (
    METACOG_BINS,
    METACOG_CURIOUS_SCALE,
    METACOG_FILE,
    METACOG_MIN_SAMPLES,
    METACOG_RETREAT_SCALE,
    METACOG_SAMPLES,
    METACOG_SHRINK_BAD,
    METACOG_SHRINK_WEAK,
    METACOG_SURPRISE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def conf_bin(conf: float) -> int:
    """Bucket confidence [0, 1] into ``METACOG_BINS`` coarse bins."""
    return max(0, min(METACOG_BINS - 1, int(float(conf) * METACOG_BINS)))


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation between two equal-length samples."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (math.sqrt(vx) * math.sqrt(vy))


class MetaMonitor:
    """Second-order confidence: how reliable is my current confidence?"""

    _path: Path | None

    def __init__(self, filepath: Path | None = None, persist: bool = True) -> None:
        """Start empty or restore a previously persisted calibration window."""
        self._samples: deque[tuple[int, int]] = deque(maxlen=METACOG_SAMPLES)
        if filepath is not None:
            self._path = Path(filepath)
        elif persist:
            env_file = os.environ.get("HANOO_METACOG_FILE", "").strip()
            self._path = Path(env_file) if env_file else METACOG_FILE
        else:
            self._path = None
        if self._path is not None and self._path.exists():
            try:
                self.load(json.loads(self._path.read_text()))
            except (OSError, ValueError, TypeError):
                logger.warning("Metacog state unreadable; starting empty.")

    def update(self, conf: float, won: bool) -> None:
        """Record one (confidence bin, outcome) calibration pair."""
        self._samples.append((conf_bin(conf), 1 if won else 0))
        self._persist()

    def reliability(self) -> float:
        """Rolling calibration correlation, normalized to [0, 1]."""
        if len(self._samples) < METACOG_MIN_SAMPLES:
            return 1.0
        xs = [float(s[0]) for s in self._samples]
        ys = [float(s[1]) for s in self._samples]
        corr = _pearson(xs, ys)
        return float(max(0.0, min(1.0, 0.5 + 0.5 * corr)))

    def sizing_scalar(self) -> float:
        """Shrink confidence-based sizing as the calibration degrades."""
        rel = self.reliability()
        if rel >= 0.6:
            return 1.0
        if rel >= 0.4:
            return METACOG_SHRINK_WEAK
        return METACOG_SHRINK_BAD

    def surprise(self, alpha: dict[str, float], episodic: Any) -> float:
        """Novelty: how far is this situation from any known pattern?"""
        _, sim = episodic.predict(alpha)
        if float(sim) <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - float(sim))))

    def curiosity_scale(self, surprise: float, pillar_state: str) -> float:
        """Explore a novel situation when stable; retreat when falling."""
        if surprise < METACOG_SURPRISE_THRESHOLD:
            return 1.0
        if pillar_state in ("tipping", "fallen"):
            return METACOG_RETREAT_SCALE
        return METACOG_CURIOUS_SCALE

    def clear(self) -> None:
        """Reset the calibration window (ironclade cleanup)."""
        self._samples.clear()
        self._persist()

    def save(self) -> dict[str, Any]:
        """Serialize the calibration window to a JSON-able dict."""
        return {"samples": list(self._samples)}

    def load(self, data: dict[str, Any]) -> None:
        """Rebuild the window from a ``save()`` payload."""
        self._samples.clear()
        for pair in data.get("samples", []):
            self._samples.append((int(pair[0]), int(pair[1])))

    def _persist(self) -> None:
        """Atomically write the calibration window when persistence is on."""
        if self._path is None:
            return
        payload = json.dumps(self.save(), indent=2)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(payload)
        tmp.replace(self._path)

    @property
    def size(self) -> int:
        """Number of recorded calibration pairs."""
        return len(self._samples)


__all__ = ["MetaMonitor", "conf_bin"]
