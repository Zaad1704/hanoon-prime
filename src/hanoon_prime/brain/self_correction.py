"""DNN gatekeeper self-review: ledger, calibration, review queue."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CAL_WINDOW, ABSTAIN_P_WIN = 200, 0.50


def _default_dir() -> Path:
    """Ledger dir (SELF_CORRECTION_DIR overrides the default)."""
    env = os.environ.get("SELF_CORRECTION_DIR")
    base = Path(__file__).resolve().parents[3] / "runtime" / "self_correction"
    return Path(env) if env else base


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    """Best-effort JSON write; logs but never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
    except OSError as exc:
        log.warning("self-correction write failed: %s", exc)


def _append(path: Path, record: dict[str, Any]) -> bool:
    """Append one JSON record; False when the write fails."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return True
    except OSError as exc:
        log.warning("self-correction write failed: %s", exc)
        return False


def _records(path: Path) -> list[dict[str, Any]]:
    """Read JSON-lines records; corrupt lines are skipped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


class PredictionLedger:
    """Append-only log of gate decisions and their resolutions."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self._dir = Path(directory) if directory is not None else _default_dir()
        self._file = self._dir / "predictions.jsonl"

    def log_decision(
        self,
        *,
        ticker: str = "",
        p_win: float,
        features: list[float],
        size_scale: float,
        admitted: bool,
        abstained: bool,
        weight: float,
        drift_z: float,
    ) -> str:
        """Append one gate decision; returns the prediction id."""
        pred_id = uuid.uuid4().hex[:16]
        feats = [round(float(v), 4) for v in features]
        digest = hashlib.sha256(
            ",".join(f"{v:.4f}" for v in feats).encode()
        ).hexdigest()[:16]
        _append(
            self._file,
            {
                "id": pred_id,
                "ts": time.time(),
                "type": "open",
                "ticker": ticker,
                "p_win": round(p_win, 4),
                "predicted_edge": round(p_win - 0.5, 4),
                "features_hash": digest,
                "features": feats,
                "size_scale": round(size_scale, 4),
                "admitted": bool(admitted),
                "abstained": bool(abstained),
                "weight": weight,
                "drift_z": round(drift_z, 3),
            },
        )
        return pred_id

    def resolve_latest(
        self,
        *,
        ticker: str,
        realized_pnl: float,
        won: bool,
        hold_minutes: float | None = None,
    ) -> str | None:
        """Resolve the newest still-open prediction for a ticker."""
        from .self_correction_policy import CorrectionJournal

        records = _records(self._file)
        closed = {r.get("id") for r in records if r.get("type") == "resolve"}
        rec = next(
            (
                r
                for r in reversed(records)
                if r.get("type") == "open"
                and r.get("id") not in closed
                and r.get("ticker", "") in ("", ticker)
            ),
            None,
        )
        if rec is None:
            return None
        _append(
            self._file,
            {
                "id": rec["id"],
                "ts": time.time(),
                "type": "resolve",
                "ticker": ticker,
                "realized_pnl": round(realized_pnl, 4),
                "won": bool(won),
                "hold_minutes": hold_minutes,
            },
        )
        CorrectionJournal(self._dir).maybe_enqueue(
            pred_id=str(rec["id"]),
            p_win=float(rec.get("p_win", 0.5)),
            features=[float(v) for v in rec.get("features", [])],
            realized_pnl=realized_pnl,
            won=won,
            ticker=ticker,
        )
        return str(rec["id"])

    def calibration_data(
        self, n: int = CAL_WINDOW
    ) -> tuple[list[tuple[float, bool]], int]:
        """Newest-first (p_win, won) pairs plus the resolve count."""
        opens: dict[str, float] = {}
        pairs: list[tuple[float, bool]] = []
        total = 0
        for r in _records(self._file):
            if r.get("type") == "open":
                opens[str(r.get("id"))] = float(r.get("p_win", 0.5))
            elif r.get("type") == "resolve":
                total += 1
                p = opens.pop(str(r.get("id")), None)
                if p is not None:
                    pairs.append((p, bool(r.get("won"))))
        return list(reversed(pairs[-n:])), total


class CalibrationMonitor:
    """Rolling calibration metrics over resolved predictions."""

    @staticmethod
    def summarize(pairs: list[tuple[float, bool]]) -> dict[str, float]:
        """Accuracy, Brier score, slope, and overconfidence gap."""
        n = len(pairs)
        if not n:
            return {"n": 0.0, "accuracy": 0.0, "brier": 1.0, "slope": 0.0, "gap": 0.0}
        ps = [p for p, _ in pairs]
        ys = [1.0 if w else 0.0 for _, w in pairs]
        mp, my = sum(ps) / n, sum(ys) / n
        var = sum((p - mp) ** 2 for p in ps)
        cov = sum((p - mp) * (y - my) for p, y in zip(ps, ys))
        hits = sum(1 for p, y in zip(ps, ys) if (p >= 0.5) == (y == 1.0))
        return {
            "n": float(n),
            "accuracy": hits / n,
            "brier": sum((p - y) ** 2 for p, y in zip(ps, ys)) / n,
            "slope": cov / var if var >= 1e-12 else 0.0,
            "gap": mp - my,
        }
