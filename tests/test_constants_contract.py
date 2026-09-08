"""tests/test_constants_contract.py — M0: constants-vs-documents drift guard.

The single class of bug that made hanoon_rebuild un-debuggable is "the
docstring says 0.18/0.35, the code says 0.25/0.55." These tests assert
that every numeric guarantee the codebase advertises equals the LIVE
constant, and that no source file advertises a stale decimal.

Tagged @contract — always run in CI, cannot be skipped.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── The immune literals: source of truth, cannot be weakened ──────────
def test_immune_literals_locked():
    """R6: every safety limit is a literal. Assert the exact values so a
    stray edit can never silently change risk parameters."""
    from hanoon_prime.immune import (
        ATR_STOP_MULT,
        ATR_TARGET_MULT,
        CONSECUTIVE_LOSSES_PAUSE,
        DAILY_LOSS_LIMIT,
        FEE_RATE,
        FIXED_FEE,
        INDICATOR_NAMES,
        KELLY_FRACTION,
        MAX_CONCURRENT_POSITIONS,
        MAX_LOSS_PER_TRADE,
        MAX_POSITION_NOTIONAL,
        PRIOR_BOTTOM,
        PRIOR_TOP,
        PRIOR_TOP_MAX,
        SCORE_INVERT,
        TARGET_R_R,
        Z_CLIP,
        Z_NORM_WINDOW,
    )

    assert (PRIOR_BOTTOM, PRIOR_TOP, PRIOR_TOP_MAX) == (0.25, 0.60, 0.65)
    assert SCORE_INVERT is False
    assert MAX_POSITION_NOTIONAL == 5_000.0
    assert MAX_LOSS_PER_TRADE == 50.0
    assert MAX_CONCURRENT_POSITIONS == 3
    assert DAILY_LOSS_LIMIT == 200.0
    assert CONSECUTIVE_LOSSES_PAUSE == 3
    assert KELLY_FRACTION == 0.25
    assert TARGET_R_R == 3.0
    assert ATR_STOP_MULT == 2.0
    assert ATR_TARGET_MULT == 6.0
    assert Z_CLIP == 3.0
    assert Z_NORM_WINDOW == 50
    assert len(INDICATOR_NAMES) == 5


# ── The live edge math MUST match the documented band ──────────────────
def test_edge_win_prob_band_matches_prior():
    """score_to_win_prob must map |score|∈[0,1] → [PRIOR_BOTTOM, PRIOR_TOP].
    This FAILS if the function drifts from the immune constants — the
    exact trap that hid rebuild's stale 0.18/0.35 docstring."""
    from hanoon_prime.edge import score_to_win_prob
    from hanoon_prime.immune import PRIOR_BOTTOM, PRIOR_TOP

    span = PRIOR_TOP - PRIOR_BOTTOM
    assert score_to_win_prob(0.0) == pytest.approx(PRIOR_BOTTOM)
    assert score_to_win_prob(1.0) == pytest.approx(PRIOR_TOP)
    assert score_to_win_prob(-1.0) == pytest.approx(PRIOR_TOP)  # R5: directionless
    assert score_to_win_prob(0.5) == pytest.approx(PRIOR_BOTTOM + 0.5 * span)
    assert score_to_win_prob(2.0) == pytest.approx(PRIOR_TOP)  # clamped
    assert score_to_win_prob(-2.0) == pytest.approx(PRIOR_TOP)


# ── No stale numeric band may be advertised in DOCSTRINGS ──────────────
# Rebuild's edge.py docstring claimed "0.18/0.35" while its constants.py said
# 0.25/0.55. Note: 0.18/0.35 also appear LEGITIMATELY as exit-threshold
# defaults in adaptive_thresholds.DEFAULTS (exit_base/exit_scale) — so we scan
# DOCSTRINGS ONLY, never code. A docstring that restates a stale band is the
# lie we're forbidding; a constant literal in code is fine.
_STALE_BANDS = ["0.18", "0.35"]


def _all_docstrings() -> list[tuple[str, str]]:
    """Return (module_or_func_name, docstring) for every docstring in src/."""
    import ast

    out: list[tuple[str, str]] = []
    for path in (SRC / "hanoon_prime").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            ds = ast.get_docstring(node)
            if ds:
                out.append((f"{path.name}:{getattr(node, 'name', 'module')}", ds))
    return out


def test_no_stale_prior_band_literals_in_docstrings():
    """No *docstring* may restate the stale 0.18/0.35 win-prob band.
    (Code literals like exit_base=0.18 are allowed — we only forbid the
    lie in prose, where it can't be asserted by a test.)"""
    offenders = []
    for where, doc in _all_docstrings():
        for stale in _STALE_BANDS:
            if stale in doc:
                offenders.append(f"{where}: '{stale}' in docstring")
    assert not offenders, f"Stale prior-band literal in docstring: {offenders}"


def test_docstrings_cite_constants_not_bare_thresholds():
    """Any docstring describing a threshold must reference the live
    constant by name, not restate a bare decimal (which then drifts)."""
    import hanoon_prime.edge as edge

    assert "PRIOR_BOTTOM" in edge.__doc__ and "PRIOR_TOP" in edge.__doc__
    import hanoon_prime.immune as imm

    # immune docstring must reference the R-rules, not magic numbers
    assert "R6" in imm.__doc__


# ── Dynamic PRIOR_TOP: immune-bounded, cold-start, faithful formula ──────
def test_dynamic_prior_top_constants_and_formula():
    """DYNAMIC_PRIOR_TOP_* must be R6 literals; the ceiling == PRIOR_TOP_MAX
    (R5 runtime guard); the formula earns wins UP / losses DOWN and cold-
    starts on the static PRIOR_TOP."""
    from hanoon_prime.edge import get_dynamic_prior_top, score_to_win_prob
    from hanoon_prime.immune import (
        DYNAMIC_PRIOR_TOP_BLEND,
        DYNAMIC_PRIOR_TOP_ENABLED,
        DYNAMIC_PRIOR_TOP_MAX,
        DYNAMIC_PRIOR_TOP_MIN,
        DYNAMIC_PRIOR_TOP_MIN_TRADES,
        DYNAMIC_PRIOR_TOP_SCALE,
        PRIOR_BOTTOM,
        PRIOR_TOP,
        PRIOR_TOP_MAX,
    )

    # R6 literals + ceiling is exactly PRIOR_TOP_MAX (R5 runtime guard).
    assert DYNAMIC_PRIOR_TOP_ENABLED is True
    assert DYNAMIC_PRIOR_TOP_MIN == 0.35
    assert DYNAMIC_PRIOR_TOP_MAX == 0.65
    assert DYNAMIC_PRIOR_TOP_MAX == PRIOR_TOP_MAX
    assert DYNAMIC_PRIOR_TOP_SCALE == 0.6
    assert DYNAMIC_PRIOR_TOP_MIN_TRADES == 20
    assert DYNAMIC_PRIOR_TOP_BLEND == 0.7

    # Cold start: not enough trades ⇒ structural static PRIOR_TOP.
    assert get_dynamic_prior_top(0.99, 0) == PRIOR_TOP
    assert get_dynamic_prior_top(0.99, DYNAMIC_PRIOR_TOP_MIN_TRADES - 1) == PRIOR_TOP

    # WR=0.45 reference ⇒ blended == static PRIOR_TOP (no movement).
    assert get_dynamic_prior_top(0.45, DYNAMIC_PRIOR_TOP_MIN_TRADES) == pytest.approx(
        PRIOR_TOP
    )

    # Winning streak earns a HIGHER cap, but never past PRIOR_TOP_MAX (R5).
    hot = get_dynamic_prior_top(0.95, 50)
    assert hot > PRIOR_TOP
    assert hot <= PRIOR_TOP_MAX

    # Losing streak tightens the cap below the static PRIOR_TOP.
    cold = get_dynamic_prior_top(0.10, 50)
    assert cold < PRIOR_TOP
    assert cold >= DYNAMIC_PRIOR_TOP_MIN

    # score_to_win_prob widens its output band to the dynamic cap, but the
    # # output never exceeds the cap (R5: win_prob ≤ 0.65 at score=1).
    wp = score_to_win_prob(1.0, prior_top=hot)
    assert PRIOR_BOTTOM <= wp <= hot
    assert wp <= PRIOR_TOP_MAX


# ── Off-by-default safety / opt-in flags (Aegis doctrine) ───────────
def test_safety_opt_in_flags_default_off():
    """R6 literals: death-spiral PROBE, contrarian override, and the
    calibration nudge are all OFF until a human opts in for production.
    A single careless flip must fail this test."""
    from hanoon_prime.immune import (
        CALIBRATION_NUDGE_ENABLED,
        CONTRARIAN_MODE_ENABLED,
        PROBE_RECOVERY_ENABLED,
    )

    assert PROBE_RECOVERY_ENABLED is False
    assert CONTRARIAN_MODE_ENABLED is False
    assert CALIBRATION_NUDGE_ENABLED is False


def test_calibration_bound_matches_rebuild():
    """Faithful port: ±0.10 (rebuild CALIB_BOUND / SCORE_CALIB_BOUND)."""
    from hanoon_prime.immune import CALIB_BOUND

    assert CALIB_BOUND == 0.10
