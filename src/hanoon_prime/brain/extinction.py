"""hanoon_prime.brain.extinction — context-gated inhibition on episodic recall.

Bouton's context-dependent extinction applied to the k-NN memory: every
pattern signature gets tagged with the context it was learned in (regime,
confidence bucket, horizon), and a pattern whose recent outcomes inside a
context degrade grows an inhibition weight. At retrieval the net modifier
is excitation − inhibition (both context-gated); a regime re-entry resets
the inhibition for that regime (renewal — the original trace revives when
its place returns). Persists to ``runtime/juli_extinction.json``
(``HANOO_EXTINCTION_FILE`` override).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import EPISODIC_KEYS
from .learning_config import (
    EXTINCT_CELLS_MAX,
    EXTINCT_DECAY,
    EXTINCT_FILE,
    EXTINCT_LOSS_BELOW,
    EXTINCT_MAX,
    EXTINCT_MIN_PATTERNS,
    EXTINCT_OVERLAP,
    EXTINCT_PERF_ALPHA,
    EXTINCT_STEP,
)

logger = logging.getLogger(__name__)


def conf_bin_label(conf: float) -> str:
    """Bucket a confidence into a coarse (low/mid/high) context tag."""
    c = float(conf)
    return "low" if c < 0.55 else "high" if c > 0.75 else "mid"


def context_key(regime: str, conf: float | str, horizon: str) -> str:
    """Canonical context tag: regime|confidence-bucket|horizon."""
    cb = conf if isinstance(conf, str) else conf_bin_label(float(conf))
    return f"{regime}|{cb}|{horizon}"


def _signature(alpha: dict[str, float]) -> frozenset[tuple[int, float]]:
    """Dimension-bin tags (also the cell key) for one alpha vector."""
    key = tuple(round(float(alpha.get(k, 0.5)), 1) for k in EPISODIC_KEYS)
    return frozenset((i, key[i]) for i in range(len(key)))


class ExtinctionTracker:
    """Per-context inhibition weights over episodic pattern signatures."""

    _path: Path | None

    def __init__(self, filepath: Path | None = None, persist: bool = True) -> None:
        """Start empty or restore a previously persisted extinction state."""
        self._cells: dict[tuple[frozenset[tuple[int, float]], str], dict[str, Any]] = {}
        if filepath is not None:
            self._path = Path(filepath)
        elif persist:
            env_file = os.environ.get("HANOO_EXTINCTION_FILE", "").strip()
            self._path = Path(env_file) if env_file else EXTINCT_FILE
        else:
            self._path = None
        if self._path is not None and self._path.exists():
            try:
                self.load(json.loads(self._path.read_text()))
            except (OSError, ValueError, TypeError):
                logger.warning("Extinction state unreadable; starting empty.")

    def clear(self) -> None:
        """Reset all context cells and persist the cleared state."""
        self._cells.clear()
        self._persist()

    def record(
        self,
        alpha: dict[str, float],
        outcome: float,
        regime: str,
        conf: float | str,
        horizon: str,
    ) -> float:
        """Feed one outcome into the pattern's context cell; return its inhibition."""
        tags = _signature(alpha)
        ck = context_key(regime, conf, horizon)
        cell = self._cells.setdefault(
            (tags, ck),
            {"patterns": 0, "perf": 0.0, "inhibition": 0.0, "tags": tags},
        )
        cell["patterns"] += 1
        cell["perf"] = (
            cell["perf"] * (1 - EXTINCT_PERF_ALPHA) + outcome * EXTINCT_PERF_ALPHA
        )
        if (
            cell["patterns"] >= EXTINCT_MIN_PATTERNS
            and cell["perf"] < EXTINCT_LOSS_BELOW
        ):
            cell["inhibition"] = min(EXTINCT_MAX, cell["inhibition"] + EXTINCT_STEP)
        else:
            cell["inhibition"] = max(0.0, cell["inhibition"] - EXTINCT_DECAY)
        self._enforce_capacity()
        self._persist()
        return float(cell["inhibition"])

    def inhibition(
        self,
        alpha: dict[str, float],
        regime: str,
        conf: float | str,
        horizon: str,
    ) -> float:
        """Context-gated inhibition for a query, from overlapping neighbor cells."""
        tags = _signature(alpha)
        ck = context_key(regime, conf, horizon)
        best = 0.0
        for (_, ctx), cell in self._cells.items():
            if ctx != ck:
                continue
            shared = len(tags & cell["tags"])
            if shared < EXTINCT_OVERLAP:
                continue
            best = max(best, cell["inhibition"] * shared / len(tags))
        return best

    def reactivate(self, regime: str) -> int:
        """Reset inhibition for cells tagged in ``regime`` (Bouton renewal)."""
        cleared = 0
        for (_, ctx), cell in self._cells.items():
            if cell["inhibition"] > 0 and ctx.startswith(f"{regime}|"):
                cell["inhibition"] = 0.0
                cleared += 1
        if cleared:
            self._persist()
        return cleared

    def _enforce_capacity(self) -> None:
        """Drop the coldest cells when the signature map grows unbounded."""
        if len(self._cells) <= EXTINCT_CELLS_MAX:
            return
        coldest = sorted(
            self._cells.keys(),
            key=lambda key: (
                self._cells[key]["patterns"],
                self._cells[key]["inhibition"],
            ),
        )
        for key in coldest[: len(self._cells) - EXTINCT_CELLS_MAX]:
            del self._cells[key]

    def save(self) -> dict[str, Any]:
        """Serialize the extinction state to a JSON-able dict."""
        return {
            "cells": [
                {
                    "context": key[1],
                    "patterns": cell["patterns"],
                    "perf": cell["perf"],
                    "inhibition": cell["inhibition"],
                    "tags": sorted(list(t) for t in cell["tags"]),
                }
                for key, cell in self._cells.items()
            ]
        }

    def load(self, data: dict[str, Any]) -> None:
        """Rebuild cells from a ``save()`` payload."""
        self._cells = {}
        for entry in data.get("cells", []):
            tags = frozenset((int(a), round(float(b), 1)) for a, b in entry["tags"])
            key = (tags, entry["context"])
            self._cells[key] = {
                "patterns": int(entry["patterns"]),
                "perf": float(entry["perf"]),
                "inhibition": float(entry["inhibition"]),
                "tags": tags,
            }

    def _persist(self) -> None:
        """Atomically write the extinction state when persistence is on."""
        if self._path is None:
            return
        payload = json.dumps(self.save(), indent=2)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(payload)
        tmp.replace(self._path)

    @property
    def size(self) -> int:
        """Number of tracked context cells."""
        return len(self._cells)
