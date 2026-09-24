"""Correction journal: review queue for high-confidence mistakes."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .self_correction import _append, _default_dir

log = logging.getLogger(__name__)

REVIEW_P_HIGH, REVIEW_P_LOW = 0.70, 0.30
REVIEW_PNL_LOSS, REVIEW_PNL_WIN, REVIEW_MAX_DRAIN = -0.01, 0.01, 32


class CorrectionJournal:
    """Review queue for high-confidence mistakes (read-only)."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self._dir = Path(directory) if directory is not None else _default_dir()
        self._file = self._dir / "review_queue.jsonl"

    def maybe_enqueue(
        self,
        *,
        pred_id: str,
        p_win: float,
        features: list[float],
        realized_pnl: float,
        won: bool,
        ticker: str,
    ) -> bool:
        """Queue confident-win-that-lost or confident-loss-that-won."""
        if not (
            (p_win >= REVIEW_P_HIGH and realized_pnl <= REVIEW_PNL_LOSS)
            or (p_win <= REVIEW_P_LOW and realized_pnl >= REVIEW_PNL_WIN)
        ):
            return False
        return _append(
            self._file,
            {
                "pred_id": pred_id,
                "ts": time.time(),
                "ticker": ticker,
                "p_win": round(p_win, 4),
                "realized_pnl": round(realized_pnl, 4),
                "won": bool(won),
                "features": [round(float(v), 4) for v in features],
            },
        )

    def drain(self) -> list[dict[str, Any]]:
        """Move queued items to reviewed.jsonl; returns the items."""
        try:
            lines = self._file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        items: list[dict[str, Any]] = []
        for line in lines[:REVIEW_MAX_DRAIN]:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if _append(
                self._dir / "reviewed.jsonl", {**item, "reviewed_ts": time.time()}
            ):
                items.append(item)
        rest = lines[REVIEW_MAX_DRAIN:]
        try:
            self._file.write_text(
                "\n".join(rest) + ("\n" if rest else ""), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("correction journal drain failed: %s", exc)
        return items


from .self_correction_policy import (
    DeratingPolicy,
    apply_derating,
    evaluate_weight,
    request_retrain,
)

__all__ = [
    "CorrectionJournal",
    "DeratingPolicy",
    "REVIEW_MAX_DRAIN",
    "REVIEW_P_HIGH",
    "REVIEW_P_LOW",
    "REVIEW_PNL_LOSS",
    "REVIEW_PNL_WIN",
    "apply_derating",
    "evaluate_weight",
    "request_retrain",
]
