"""brain.realized_ev — Realized trade statistics store (rebuild ev_gate.py port).

Persists band WR, confidence-bin WR and realized R:R from closed real
trades. Gate math lives in ``brain/ev_gate.py``; re-exported here so
``from .realized_ev import ev_gate_should_enter`` keeps working.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from ..edge import get_dynamic_prior_top, score_to_win_prob
from ..immune import CALIB_BOUND, CALIBRATION_NUDGE_ENABLED, TARGET_R_R
from .config import (
    BAND_MIN_SAMPLES,
    CONF_MIN_SAMPLES,
    JULI_REALIZED_FILE,
    NASH_GATE_AUTHORITY_WR,
    RR_MAX_LOOKBACK,
    RR_MIN_TRADES,
)

log = logging.getLogger(__name__)
_BAND_COUNT: int = 10

# Confidence bins (v2.1): realized WR per JULI-confidence band; a
# proven-losing confidence band drags EV below the floor and the gate
# refuses; thin bins fall back to the structural prior.
CONF_BIN_COUNT: int = 10  # bin 0 = conf 0.50, bin 9 = conf 0.95

# Aggressive learning: confidence bins below CONF_MIN_SAMPLES normally
# don't influence the gate. But a bin with >= this many losses AND zero
# wins is an active money-burner — trigger emergency threshold tightening
# immediately rather than waiting for CONF_MIN_SAMPLES (20) to accumulate.
CONF_LOSS_STREAK_WARN: int = 10

# Per loss in a losing bin, raise the entry threshold by this much.
CONF_LOSS_AGGRESSIVE_STEP: float = 0.025

# Cap on how aggressively a single trade-close can raise the threshold
# when losing conf bins are detected.
CONF_LOSS_AGGRESSIVE_MAX: float = 0.15


def _conf_bin(conf: float) -> int:
    """Map confidence [0.50, 1.0] to a bin index (0..CONF_BIN_COUNT-1)."""
    c = min(1.0, max(0.50, float(conf)))
    idx = int((c - 0.50) / 0.05)
    return min(CONF_BIN_COUNT - 1, max(0, idx))


def _band_key(score: float) -> int:
    """Map a score magnitude to its band index (0..9)."""
    magnitude = min(1.0, max(0.0, abs(float(score))))
    return int(round(magnitude * (_BAND_COUNT - 1)))


class RealizedStats:
    """In-memory + persisted realized trade statistics (band WR + RR)."""

    _path: Optional[Path]

    def __init__(self, filepath: Optional[Path] = None, persist: bool = True) -> None:
        """Construct realized stats, optionally loading from ``filepath``."""
        if filepath is not None:
            self._path = Path(filepath)
        elif persist:
            env_file = os.environ.get("HANOO_REALIZED_FILE", "").strip()
            self._path = Path(env_file) if env_file else JULI_REALIZED_FILE
        else:
            self._path = None
        self._lock = threading.RLock()
        self._band_wins: dict[int, int] = defaultdict(int)
        self._band_losses: dict[int, int] = defaultdict(int)
        self._conf_wins: dict[int, int] = defaultdict(int)
        self._conf_losses: dict[int, int] = defaultdict(int)
        self._rr: deque[tuple[int, float, int]] = deque(maxlen=RR_MAX_LOOKBACK)
        if self._path is not None:
            self._load()

    def reset(self) -> None:
        """Clear all stats and delete persisted file (ironclade cleanup)."""
        with self._lock:
            self._band_wins = defaultdict(int)
            self._band_losses = defaultdict(int)
            self._conf_wins = defaultdict(int)
            self._conf_losses = defaultdict(int)
            self._rr = deque(maxlen=RR_MAX_LOOKBACK)
        if self._path is not None and self._path.exists():
            self._path.unlink()

    def add_outcome(
        self, score: float, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Record a closed *real* trade and refresh band/RR stats."""
        band = _band_key(score)
        with self._lock:
            if won:
                self._band_wins[band] += 1
            else:
                self._band_losses[band] += 1
            self._rr.append((1 if won else 0, abs(float(pnl_pct)), int(direction)))
        self._save()

    def add_confidence_outcome(self, conf: float, won: bool) -> None:
        """Record a real trade outcome into its confidence bin."""
        b = _conf_bin(conf)
        with self._lock:
            if won:
                self._conf_wins[b] += 1
            else:
                self._conf_losses[b] += 1
        self._save()

    def band_wr(self, score: float) -> tuple[float, float, int]:
        """Return (win_rate, reliability, n) for the band of ``score``."""
        band = _band_key(score)
        with self._lock:
            wins = self._band_wins.get(band, 0)
            losses = self._band_losses.get(band, 0)
        n = wins + losses
        if n == 0:
            return 0.5, 0.0, 0
        return wins / n, min(1.0, n / (n + BAND_MIN_SAMPLES)), n

    def realized_rr(self) -> tuple[float, float, int]:
        """Return (realized_RR, reliability, n) over closed trades."""
        with self._lock:
            won_pnl = [p for f, p, _d in self._rr if f]
            lost_pnl = [p for f, p, _d in self._rr if not f]
            n = len(self._rr)
        if n == 0:
            return float(TARGET_R_R), 0.0, 0
        avg_win = sum(won_pnl) / len(won_pnl) if won_pnl else 0.0
        avg_loss = sum(lost_pnl) / len(lost_pnl) if lost_pnl else 0.0
        rr = avg_win / avg_loss if avg_loss > 0 else float(TARGET_R_R)
        return rr, min(1.0, n / (n + RR_MIN_TRADES)), n

    def is_gate_closed(self, score: float) -> bool:
        """True when the score band is a statistically losing region."""
        wr, _rel, n = self.band_wr(score)
        return bool(n >= BAND_MIN_SAMPLES and wr < NASH_GATE_AUTHORITY_WR)

    def conf_band_wr(self, conf: float) -> tuple[float, float, int]:
        """Return (win_rate, reliability, n) for a confidence bin.

        Thin data threshold: only return actual WR when n >= CONF_MIN_SAMPLES.
        Otherwise fall back to 0.5 (neutral) with reliability 0.0.
        """
        b = _conf_bin(conf)
        with self._lock:
            wins = self._conf_wins.get(b, 0)
            losses = self._conf_losses.get(b, 0)
        n = wins + losses
        # Thin data: fall back to structural prior (0.5 WR, 0.0 reliability)
        if n < CONF_MIN_SAMPLES:
            return 0.5, 0.0, 0
        return wins / n, min(1.0, n / (n + CONF_MIN_SAMPLES)), n

    @property
    def conf_bin_count(self) -> int:
        """Number of populated confidence bins (telemetry)."""
        with self._lock:
            keys = set(self._conf_wins) | set(self._conf_losses)
            return len(keys)

    @property
    def total_trades(self) -> int:
        """Total number of recorded real trades."""
        with self._lock:
            return sum(self._band_wins.values()) + sum(self._band_losses.values())

    def recent_win_rate(self) -> tuple[float, int]:
        """Overall recent realized win rate and trade count, cold-safe.

        Folded from the realized-R:R ring (win flags) — mirrors rebuild
        ``memory._recent_wr`` without a separate deque. Empty store returns
        (0.5, 0) so callers fall back to the structural prior, never a crash.
        """
        with self._lock:
            samples = list(self._rr)
        if not samples:
            return 0.5, 0
        wins = sum(1 for f, _p, _d in samples if f)
        return wins / len(samples), len(samples)

    def losing_conf_bins(self) -> list[tuple[int, int, int]]:
        """Return bins with 0 wins and >= CONF_LOSS_STREAK_WARN losses.

        Each entry is ``(bin_index, losses, wins)``. These are confidence
        bands that are actively losing money — the aggressive learning
        hook in Dynamics reads these to accelerate threshold adaptation
        rather than waiting for CONF_MIN_SAMPLES (20).
        """
        result: list[tuple[int, int, int]] = []
        with self._lock:
            for b in sorted(set(self._conf_losses)):
                wins = self._conf_wins.get(b, 0)
                losses = self._conf_losses.get(b, 0)
                if wins == 0 and losses >= CONF_LOSS_STREAK_WARN:
                    result.append((b, losses, wins))
        return result

    def dynamic_prior_top(self) -> float:
        """Dynamic PRIOR_TOP for the brain's confidence cap (see edge).

        Cold-start returns the static PRIOR_TOP until
        DYNAMIC_PRIOR_TOP_MIN_TRADES real trades accumulate. Hard-clamped
        to [DYNAMIC_PRIOR_TOP_MIN, DYNAMIC_PRIOR_TOP_MAX] whose ceiling is
        PRIOR_TOP_MAX, so the entry gate (R5) never over-believes past 0.65.
        """
        wr, n = self.recent_win_rate()
        return get_dynamic_prior_top(wr, n)

    def calibration_adjustment(self, score: float) -> float:
        """Faithful port of rebuild ``prediction_error_adjustment``.

        Returns ``realized_band_wr - predicted_win_prob``, clamped to
        ±``CALIB_BOUND``. Positive => the score band beat its prediction
        (lean in); negative => it underperformed (pull back). When the flag
        is off or data is thin, returns 0.0 (no-op) so live behavior is
        unchanged until opted in.

        Prime mapping: rebuild's ``predicted_win_prob`` (linear
        THRESHOLD_MIN..THRESHOLD_MAX → PRIOR_BOTTOM..PRIOR_TOP) is approximated
        by Prime's own ``score_to_win_prob``, and the band WR by
        :meth:`band_wr` (same realization source the ev_gate trusts).
        """
        if not CALIBRATION_NUDGE_ENABLED:
            return 0.0
        wr, _rel, n = self.band_wr(score)
        if n < BAND_MIN_SAMPLES:
            return 0.0
        pred_wp = score_to_win_prob(abs(score))
        adj = wr - pred_wp
        return max(-CALIB_BOUND, min(CALIB_BOUND, adj))

    def snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot of realized state."""
        with self._lock:
            return {
                "total": self.total_trades,
                "band_wins": dict(self._band_wins),
                "band_losses": dict(self._band_losses),
                "conf_wins": dict(self._conf_wins),
                "conf_losses": dict(self._conf_losses),
                "rr_samples": list(self._rr),
            }

    def _save(self) -> None:
        """Atomically write the snapshot to ``self._path`` (if enabled)."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot()), encoding="utf-8")
            tmp.replace(self._path)
        except (OSError, ValueError, TypeError) as exc:
            log.warning("RealizedStats save failed: %s", exc)

    def _load(self) -> None:
        """Restore band/conf/RR counters from disk."""
        assert self._path is not None
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._band_wins = defaultdict(
                int, {int(k): v for k, v in data.get("band_wins", {}).items()}
            )
            self._band_losses = defaultdict(
                int, {int(k): v for k, v in data.get("band_losses", {}).items()}
            )
            self._rr = deque(
                ((int(w), float(p), int(d)) for w, p, d in data.get("rr_samples", [])),
                maxlen=RR_MAX_LOOKBACK,
            )
        except (OSError, ValueError, TypeError) as exc:
            log.warning("RealizedStats load failed: %s", exc)


# Gate math lives in brain/ev_gate.py; re-exported for API stability.
from .ev_gate import compute_ev_and_entry, ev_gate_should_enter, verify_learning_gate

__all__ = [
    "RealizedStats",
    "compute_ev_and_entry",
    "ev_gate_should_enter",
    "verify_learning_gate",
]
