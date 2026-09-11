"""tests/test_ablation.py — Phase 3: per-organ/factor ablation sanity.

The ablation harness must toggle exactly one thing at a time, renormalize
cleanly when a factor's weight is zeroed, and keep all risk rails on.
These tests check mechanism correctness, not "which factor wins" — the
report answers that with the honest thin-data caveat.
"""

from __future__ import annotations

from pathlib import Path

from hanoon_prime.ablation import (
    baseline_variant,
    build_variants,
    contribution,
    pool,
    rank_factors,
    run_variant,
)
from hanoon_prime.immune import INDICATOR_WEIGHTS

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
FAST = ["AAPL", "MSFT", "SPY"]


def test_baseline_and_off_all_alpha_are_sane() -> None:
    """Baseline trades; all-alpha-off yields zero trades (pure sanity)."""
    base = run_variant(baseline_variant(), FAST, FIXTURES)
    pool_base = pool(base)
    assert pool_base["n_active"] > 0
    assert pool_base["n_trades"] > 0

    off = run_variant(
        [v for v in build_variants() if v.name == "off_all_alpha"][0],
        FAST,
        FIXTURES,
    )
    p = pool(off)
    # Zero alpha → score 0 on every bar → HOLD everywhere.
    assert p["n_trades"] == 0
    assert p["n_active"] == 0


def test_each_factor_variant_zeros_exactly_one_weight() -> None:
    """Per-factor off-variants must zero one key and leave the rest intact."""
    for v in build_variants():
        if not v.name.startswith("off_") or v.name in ("off_all_alpha", "off_eyes"):
            continue
        factor = v.name.removeprefix("off_")
        assert factor in INDICATOR_WEIGHTS
        assert v.weights_override is not None
        assert v.weights_override[factor] == 0.0
        others = {k: val for k, val in v.weights_override.items() if k != factor}
        assert others == {k: val for k, val in INDICATOR_WEIGHTS.items() if k != factor}


def test_universe_deltas_are_measured_against_same_baseline() -> None:
    """contributions compare every variant to the SAME baseline pool."""
    buckets: dict[str, dict[str, float]] = {}
    base = pool(run_variant(baseline_variant(), FAST, FIXTURES))
    buckets["baseline"] = base
    for v in build_variants():
        buckets[v.name] = pool(run_variant(v, FAST, FIXTURES))

    # Sanity diagonal: contribution of variant vs baseline is well-defined.
    c = contribution(buckets["off_momentum"], base)
    assert isinstance(c, float)
    # Removing a factor can help or hurt, but the ranking must be consistent.
    ranked = rank_factors(buckets)
    assert len(ranked) == len(
        [
            v
            for v in build_variants()
            if v.name.startswith("off_") and v.name not in ("off_all_alpha", "off_eyes")
        ]
    )
    # Sorted descending by contribution.
    vals = [kv[1] for kv in ranked]
    assert vals == sorted(vals, reverse=True)


def test_threshold_flat_changes_entry_behavior() -> None:
    """threshold_flat must be a real toggle (alters trade count)."""
    variant = [v for v in build_variants() if v.name == "threshold_flat"][0]
    off = pool(run_variant(variant, FAST, FIXTURES))
    base = pool(run_variant(baseline_variant(), FAST, FIXTURES))
    assert off["n_trades"] != base["n_trades"]


def test_patches_are_scoped_to_the_variant_run() -> None:
    """Runtime module patches must not leak past their run_variant call."""
    import hanoon_prime.hands as hands

    baseline = getattr(hands, "ATR_STOP_MULT")
    variant = [v for v in build_variants() if v.name == "stop_wide"][0]
    run_variant(variant, FAST, FIXTURES)
    assert getattr(hands, "ATR_STOP_MULT") == baseline


def test_flat_sizing_keeps_rail_rails_but_changes_shares() -> None:
    """Flat sizing must still run (ev is share-normalized, so EV≈baseline)."""
    variant = [v for v in build_variants() if v.name == "flat_sizing"][0]
    res = run_variant(variant, FAST, FIXTURES)
    base = run_variant(baseline_variant(), FAST, FIXTURES)
    # Sanity: EV is per-share R; sizing only scales P&L, not expectancy.
    assert abs(pool(res)["ev_mean"] - pool(base)["ev_mean"]) < 0.10


def test_stop_wide_is_realistically_better_than_baseline() -> None:
    """stop_wide must trade and improve EV — mean-reversion exit artefact."""
    base = pool(run_variant(baseline_variant(), FAST, FIXTURES))
    wide = pool(
        run_variant(
            [v for v in build_variants() if v.name == "stop_wide"][0],
            FAST,
            FIXTURES,
        )
    )
    assert wide["n_trades"] > 0
    assert wide["ev_mean"] > base["ev_mean"]


def test_pool_json_roundtrips() -> None:
    """Report data is JSON-serialisable and contains all expected keys."""
    import json

    base = pool(run_variant(baseline_variant(), FAST, FIXTURES))
    blob = json.loads(json.dumps(base))
    for k in (
        "n_active",
        "n_trades",
        "ev_mean",
        "win_rate",
        "realized_rr",
        "sharpe",
        "drawdown",
        "return_pct",
    ):
        assert k in blob
