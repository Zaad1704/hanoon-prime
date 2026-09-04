"""brain.regression_guard — Automated regression guards for critical invariants.

Runs invariant checks at startup and periodically. Each guard is a
self-contained test that verifies a specific fix is still in place.

Source: rebuild's regression_guard.py (simplified for Prime).
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def _guard(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    """Auto-generated docstring."""
    return {"name": name, "passed": bool(passed), "detail": detail}


def _guard_weight_integrity() -> dict[str, Any]:
    """Verify weights sum to approximately 1.0."""
    try:
        from .config import DEFAULT_WEIGHTS

        total = sum(DEFAULT_WEIGHTS.values())
        if not (0.80 <= total <= 1.20):
            return _guard(
                "weight_integrity",
                False,
                f"DEFAULT_WEIGHTS sum={total:.4f} outside [0.80, 1.20]",
            )
        return _guard("weight_integrity", True, f"sum={total:.4f}")
    except Exception as exc:
        return _guard("weight_integrity", False, f"Import error: {exc}")


def _guard_structural_prior_bounds() -> dict[str, Any]:
    """Verify PRIOR_BOTTOM < PRIOR_TOP and both in (0, 1)."""
    try:
        from ..immune import PRIOR_BOTTOM, PRIOR_TOP, TARGET_R_R

        if not (0.0 < PRIOR_BOTTOM < 1.0):
            return _guard(
                "structural_prior", False, f"PRIOR_BOTTOM={PRIOR_BOTTOM} outside (0,1)"
            )
        if not (0.0 < PRIOR_TOP < 1.0):
            return _guard(
                "structural_prior", False, f"PRIOR_TOP={PRIOR_TOP} outside (0,1)"
            )
        if PRIOR_BOTTOM >= PRIOR_TOP:
            return _guard("structural_prior", False, f"PRIOR_BOTTOM >= PRIOR_TOP")
        ev_at_top = PRIOR_TOP * TARGET_R_R - (1.0 - PRIOR_TOP)
        if ev_at_top < 0:
            return _guard(
                "structural_prior", False, f"EV@top={ev_at_top:.4f} is negative"
            )
        return _guard(
            "structural_prior",
            True,
            f"PRIOR_BOTTOM={PRIOR_BOTTOM}, PRIOR_TOP={PRIOR_TOP}",
        )
    except Exception as exc:
        return _guard("structural_prior", False, f"Import error: {exc}")


def _guard_no_score_inversion() -> dict[str, Any]:
    """Verify score_to_win_prob never returns negative values."""
    try:
        from ..edge import score_to_win_prob

        for s in [0.0, 0.1, 0.5, 0.9, 1.0]:
            wp = score_to_win_prob(s)
            if wp < 0:
                return _guard(
                    "no_score_inversion",
                    False,
                    f"score_to_win_prob({s})={wp:.4f} is negative",
                )
        return _guard("no_score_inversion", True, "All scores map to positive WR")
    except Exception as exc:
        return _guard("no_score_inversion", False, f"Import error: {exc}")


def _guard_threshold_bounds() -> dict[str, Any]:
    """Verify threshold bounds are sensible."""
    try:
        from .config import SIGNAL_THRESHOLD, THRESHOLD_MAX, THRESHOLD_MIN

        if THRESHOLD_MIN >= THRESHOLD_MAX:
            return _guard(
                "threshold_bounds", False, f"MIN={THRESHOLD_MIN} >= MAX={THRESHOLD_MAX}"
            )
        if not (THRESHOLD_MIN <= SIGNAL_THRESHOLD <= THRESHOLD_MAX):
            return _guard(
                "threshold_bounds",
                False,
                f"SIGNAL={SIGNAL_THRESHOLD} outside [{THRESHOLD_MIN},{THRESHOLD_MAX}]",
            )
        return _guard("threshold_bounds", True, f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    except Exception as exc:
        return _guard("threshold_bounds", False, f"Import error: {exc}")


def _guard_modifier_bounds() -> dict[str, Any]:
    """Verify modifier bounds are positive and bounded."""
    try:
        from .config import AFFECTIVE_MOD_BOUND, EPISODIC_MOD_BOUND, HALIM_MOD_BOUND

        for name, val in [
            ("episodic", EPISODIC_MOD_BOUND),
            ("halim", HALIM_MOD_BOUND),
            ("affective", AFFECTIVE_MOD_BOUND),
        ]:
            if val <= 0 or val > 0.5:
                return _guard(
                    "modifier_bounds", False, f"{name}={val} outside (0, 0.5)"
                )
        return _guard("modifier_bounds", True, "All modifiers bounded")
    except Exception as exc:
        return _guard("modifier_bounds", False, f"Import error: {exc}")


ALL_GUARDS = [
    _guard_weight_integrity,
    _guard_structural_prior_bounds,
    _guard_no_score_inversion,
    _guard_threshold_bounds,
    _guard_modifier_bounds,
]


def run_all_guards() -> dict[str, Any]:
    """Run all regression guards and return summary."""
    results = []
    for guard_fn in ALL_GUARDS:
        try:
            result = guard_fn()
        except Exception as exc:
            result = _guard(guard_fn.__name__, False, f"Guard crashed: {exc}")
        results.append(result)
    failures = [r for r in results if not r["passed"]]
    return {
        "all_pass": len(failures) == 0,
        "n_pass": sum(1 for r in results if r["passed"]),
        "n_fail": len(failures),
        "guards": results,
        "failures": failures,
        "timestamp": time.time(),
    }


def run_startup_guards() -> bool:
    """Run guards at startup. Returns True if all pass."""
    result = run_all_guards()
    if result["all_pass"]:
        log.info("REGRESSION GUARD: ALL %d guards PASS", result["n_pass"])
    else:
        log.warning("REGRESSION GUARD: %d FAILURES", result["n_fail"])
        for f in result["failures"]:
            log.warning("  FAILED: %s: %s", f["name"], f["detail"])
    return bool(result["all_pass"])
