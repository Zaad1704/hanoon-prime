"""DNN derating policy and correction journal (self-correction part 2)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .self_correction import _default_dir, _save_json

log = logging.getLogger(__name__)

CAL_MIN_SAMPLES, RERATE_MIN_NEW_SAMPLES = 30, 100
DRIFT_Z_DEGRADE, DRIFT_Z_ABSTAIN = 3.0, 5.0
BRIER_HEALTHY_MAX, BRIER_CRITICAL_MIN = 0.20, 0.25
GAP_DEGRADED_MIN, GAP_CRITICAL_MIN = 0.10, 0.20
ACC_DEGRADED_MIN, ACC_CRITICAL_MIN = 0.55, 0.45
SLOPE_DEGRADED_MIN, BAR_RAISE = 0.70, 0.03


class DeratingPolicy:
    """Monotonic derating: weight only falls on bad evidence."""

    @staticmethod
    def target_weight(snapshot: dict[str, float], drift_z: float) -> tuple[float, str]:
        """Target weight in {1.0, 0.5, 0.0} with a reason."""
        brier, gap = snapshot.get("brier", 1.0), snapshot.get("gap", 0.0)
        acc, slope = snapshot.get("accuracy", 0.0), snapshot.get("slope", 0.0)
        if snapshot.get("n", 0.0) < CAL_MIN_SAMPLES:
            cal, why = 0.5, "cold start: unknown track record"
        elif (
            brier > BRIER_CRITICAL_MIN
            or gap > GAP_CRITICAL_MIN
            or acc < ACC_CRITICAL_MIN
        ):
            cal, why = 0.0, "critically miscalibrated"
        elif (
            brier > BRIER_HEALTHY_MAX
            or gap > GAP_DEGRADED_MIN
            or acc < ACC_DEGRADED_MIN
            or slope < SLOPE_DEGRADED_MIN
        ):
            cal, why = 0.5, "degraded calibration"
        else:
            cal, why = 1.0, "healthy"
        z = abs(drift_z)
        if z > DRIFT_Z_ABSTAIN:
            return 0.0, f"drift |z|={z:.1f} > {DRIFT_Z_ABSTAIN}"
        if z > DRIFT_Z_DEGRADE:
            return min(cal, 0.5), f"drift |z|={z:.1f} > {DRIFT_Z_DEGRADE}"
        return cal, why


def _rerate_step(weight: float, waited: int) -> tuple[float, str]:
    """One re-rate step up after enough new resolved samples."""
    if waited >= RERATE_MIN_NEW_SAMPLES:
        step = 0.5 if weight <= 0.0 else 1.0
        return step, "re-rated one step after 100 new resolved samples"
    return weight, "re-rate withheld: need 100 new resolved samples"


def evaluate_weight(
    snapshot: dict[str, float],
    drift_z: float,
    resolved_total: int,
    directory: Path | str | None = None,
) -> tuple[float, str]:
    """Derate immediately; re-rate one step per 100 new samples."""
    directory = Path(directory) if directory is not None else _default_dir()
    state_file = directory / "derating_state.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    weight = float(state.get("weight", 1.0))
    target, reason = DeratingPolicy.target_weight(snapshot, drift_z)
    new_weight, note = weight, "held"
    if target < weight:
        new_weight, note = target, f"derated: {reason}"
    elif target > weight:
        waited = resolved_total - int(state.get("resolved_at_change", 0))
        new_weight, note = _rerate_step(weight, waited)
    if new_weight != weight:
        _save_json(
            state_file,
            {
                "weight": new_weight,
                "resolved_at_change": resolved_total,
                "reason": note,
                "ts": time.time(),
            },
        )
        if new_weight <= 0.0:
            request_retrain(reason, snapshot, drift_z, directory)
    return new_weight, note


def request_retrain(
    reason: str,
    snapshot: dict[str, float],
    drift_z: float,
    directory: Path | str | None = None,
) -> Path | None:
    """File a PENDING_HUMAN_APPROVAL retrain request; never trains."""
    base = Path(directory) if directory is not None else _default_dir()
    out_dir = base / "retrain_requests"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path = out_dir / f"retrain_{stamp}.json"
        if not path.exists():
            path.write_text(
                json.dumps(
                    {
                        "requested_at": stamp,
                        "reason": reason,
                        "drift_z": drift_z,
                        "calibration": snapshot,
                        "status": "PENDING_HUMAN_APPROVAL",
                        "note": (
                            "No automatic retraining exists. A human must review, "
                            "retrain offline, and validate out-of-sample."
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.critical("DNN RETRAIN REQUESTED (%s): %s", reason, path)
        return path
    except OSError as exc:
        log.warning("retrain request write failed: %s", exc)
        return None


def apply_derating(
    admit: bool, p_win: float, size_scale: float, weight: float, *, threshold: float
) -> tuple[bool, float, float]:
    """Apply the derating weight; never grows size or lowers the bar."""
    bar = threshold + (BAR_RAISE if weight < 1.0 else 0.0)
    ok = weight > 0.0 and admit and p_win >= bar
    return bool(ok), p_win, round(max(0.0, min(size_scale, size_scale * weight)), 4)


from .self_correction_journal import (
    REVIEW_MAX_DRAIN,
    REVIEW_P_HIGH,
    REVIEW_P_LOW,
    REVIEW_PNL_LOSS,
    REVIEW_PNL_WIN,
    CorrectionJournal,
)

__all__ = [
    "BAR_RAISE",
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
