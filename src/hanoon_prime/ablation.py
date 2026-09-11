"""hanoon_prime.ablation — per-organ / per-factor ablation harness.

Toggles exactly ONE organ or alpha factor OFF at a time and measures the
delta against baseline on the committed fixtures. A positive factor
contribution means removing it LOWERED pooled EV (the factor adds edge);
a negative contribution means it was harmful.

Design rules (Phase 3.1):
  * One toggle per run — never simultaneous changes.
  * Saturation via weight zero: ``Cortex._tanh_score`` re-normalizes over
    present keys, so zeroing a weight removes that factor cleanly from both
    numerator and denominator.
  * All risk/slippage rails (fees, adverse fill, safety nets) stay ON.
  * Baseline brain is *fresh* per run (no z-history leakage across runs or
    variants via the weights/injected-brain path).
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

from .backtest import backtest_ticker
from .cortex import Cortex
from .eyes import load_ohlcv
from .hippocampus import Hippocampus
from .immune import INDICATOR_WEIGHTS
from .types import BarSeries

FACTOR_KEYS: tuple[str, ...] = tuple(INDICATOR_WEIGHTS)


@dataclass(frozen=True)
class Variant:
    """One ablation run: factor mask + optional module-binding patches.

    ``weights_override`` replaces the full cocktail (only safe because
    INDICATOR_WEIGHTS is a plain dict copied at Cortex construction).
    ``patches`` maps ``"module.attr"`` → value for runtime toggles that
    have no injection point (threshold, short-allowed, hands ATR params).
    """

    name: str
    label: str
    weights_override: dict[str, float] | None = None
    threshold: float | None = None
    flat_sizing: bool = False
    patches: dict[str, float | bool] = field(default_factory=dict)


def baseline_variant() -> Variant:
    """The untouched pipeline."""
    return Variant(name="baseline", label="Baseline (all organs on)")


def _factor_off(factor: str) -> Variant:
    w = dict(INDICATOR_WEIGHTS)
    w[factor] = 0.0
    return Variant(
        name=f"off_{factor}",
        label=f"Factor off: {factor}",
        weights_override=w,
    )


# Ablation specs: (name, label, kwargs-for-Variant). Kept at module level
# so ``build_variants`` stays a thin loop (R3: <=40 lines per function).
TOGGLE_SPECS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "off_all_alpha",
        "All alpha factors off",
        {"weights_override": {k: 0.0 for k in FACTOR_KEYS}},
    ),
    (
        "off_eyes",
        "Eyes features off (vpin + orderbook)",
        {
            "weights_override": {
                **INDICATOR_WEIGHTS,
                "vpin": 0.0,
                "orderbook_imbalance": 0.0,
            }
        },
    ),
    (
        "threshold_flat",
        "Cortex threshold flat (enter on any signal)",
        {"threshold": 0.0},
    ),
    (
        "short_off",
        "Short entries off",
        {"patches": {"hanoon_prime.cortex.SHORT_ALLOWED": False}},
    ),
    (
        "stop_tight",
        "ATR stop 1.0x (tighter)",
        {"patches": {"hanoon_prime.hands.ATR_STOP_MULT": 1.0}},
    ),
    (
        "stop_wide",
        "ATR stop 4.0x (wider)",
        {"patches": {"hanoon_prime.hands.ATR_STOP_MULT": 4.0}},
    ),
    (
        "target_off",
        "ATR target 60x (target kills disabled)",
        {"patches": {"hanoon_prime.hands.ATR_TARGET_MULT": 60.0}},
    ),
    ("timeout_30", "Timeout 30 bars", {}),
    ("flat_sizing", "Flat 1-share sizing", {"flat_sizing": True}),
]


def build_variants() -> list[Variant]:
    """All Phase-3 ablations (each exactly one toggle from baseline)."""
    return [_factor_off(f) for f in FACTOR_KEYS] + [
        Variant(name=n, label=l, **kw) for n, l, kw in TOGGLE_SPECS
    ]


class _FlatSizingBrain(Hippocampus):
    """Brain whose sizing always returns 1 share (sizing ablation)."""

    def size_position(
        self, _win_prob: float, _entry_price: float, _atr: float
    ) -> float:
        """Override: always size a flat 1 share (sizing ablation)."""
        return 1.0


def _make_brain(variant: Variant) -> Hippocampus:
    """Construct a fresh brain honoring the variant's toggles.

    Uses ``Cortex(threshold=...)`` so the entry threshold is injected at
    construction (it is an instance attribute, not a module global).
    Sizing ablation swaps in a flat-sizing brain. Weights are passed
    through the same path — a plain dict copied at Cortex construction.
    """
    if variant.threshold is not None:
        cortex = Cortex(weights=variant.weights_override, threshold=variant.threshold)
    else:
        cortex = Cortex(weights=variant.weights_override)
    if variant.flat_sizing:
        return _FlatSizingBrain(cortex=cortex)
    return Hippocampus(cortex=cortex)


def _patches_for(variant: Variant) -> list[Any]:
    """Component patches for a variant (module patches + timeout default)."""
    out: list[Any] = [
        mock.patch(target, value) for target, value in variant.patches.items()
    ]
    if variant.name == "timeout_30":
        import hanoon_prime.hands as hands

        out.append(mock.patch.object(hands._check_exit, "__defaults__", (30,)))
    return out


def run_variant(
    variant: Variant,
    tickers: list[str],
    data_dir: Path,
    window: int = 50,
) -> dict[str, dict[str, Any]]:
    """Run one ablation variant over the tickers; returns per-ticker metrics.

    Uses a fresh brain per variant (no z-history or weight leakage). The
    runtime module patches are scoped to the call in a context manager.
    """
    brain = _make_brain(variant)
    results: dict[str, dict[str, Any]] = {}
    with ExitStack() as cm:
        for p in _patches_for(variant):
            cm.enter_context(p)
        for t in tickers:
            path = data_dir / f"{t}_1min.csv"
            if not path.exists():
                continue
            data = load_ohlcv(path)
            results[t] = backtest_ticker(
                t,
                BarSeries(data["close"], data["high"], data["low"], data["volume"]),
                window,
                brain=brain,
            )
    return results


def pool(variant_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-ticker metrics into a universe-level pool.

    Uses the same denominator across variants for honest deltas: the ticker
    set that produced metrics (i.e., had at least one trade in the run).
    """
    active = [m for m in variant_results.values() if m["total_trades"] > 0]
    n = len(active)
    if n == 0:
        return {
            "n_active": 0.0,
            "n_trades": 0.0,
            "ev_mean": 0.0,
            "win_rate": 0.0,
            "realized_rr": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "return_pct": 0.0,
        }
    total_trades = sum(m["total_trades"] for m in active)
    # Per-ticker sharpe of a near-flat equity curve (std→0) explodes to ~1e9
    # on 5-day thin data; clip to a sane band so the report column is legible.
    sharpe_vals = [max(-20.0, min(20.0, m["sharpe_ratio"])) for m in active]
    return {
        "n_active": float(n),
        "n_trades": float(total_trades),
        "ev_mean": sum(m["ev_per_trade"] for m in active) / n,
        "win_rate": sum(m["win_rate"] for m in active) / n,
        "realized_rr": sum(m["realized_rr"] for m in active) / n,
        "sharpe": sum(sharpe_vals) / n,
        "drawdown": sum(m["max_drawdown"] for m in active) / n,
        "return_pct": sum(m["total_return_pct"] for m in active) / n,
    }


def contribution(variant_pool: dict[str, float], base_pool: dict[str, float]) -> float:
    """Marginal contribution of a toggled component (delta vs baseline EV).

    Positive = removing it hurt EV (component carries edge); negative =
    removing it helped (component was harmful); ~0 = inert.
    """
    return float(base_pool["ev_mean"] - variant_pool["ev_mean"])


def rank_factors(
    pools: dict[str, dict[str, float]],
) -> list[tuple[str, float]]:
    """Rank factor ablations by marginal contribution (descending)."""
    base = pools["baseline"]
    ranked = [
        (name, contribution(pool_row, base))
        for name, pool_row in pools.items()
        if name.startswith("off_") and name not in ("off_all_alpha", "off_eyes")
    ]
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked
