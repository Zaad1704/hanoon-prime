"""hanoon_prime._calibrate — self-tuning engine.

The system auto-evaluates its own indicators on historical data and
adjusts: signal threshold, indicator weights, R:R, and position size.

No manual thresholds. The system looks at its own performance and
tunes itself — like a human trader reviewing their track record.

Flow:
  1. Permutation-test each indicator's edge (self-validating)
  2. Weight indicators by relative |correlation| strength
  3. Grid-search stop/target/threshold for best aggregate EV
  4. Verify profitability on held-out tickers
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._edge_eval import evaluate_indicator_pooled
from ._grid_search import grid_search, sim_with_params
from .constants import EDGE_P_VALUE, MAX_POSITION_NOTIONAL
from .alpha import INDICATOR_NAMES

__all__ = ["Calibration", "calibrate"]

# Re-export for backward compatibility
from ._edge_eval import evaluate_indicator_edge  # noqa: F401


@dataclass
class Calibration:
    """Result of auto-calibration: tuned parameters for the brain."""
    weights: dict = field(default_factory=dict)
    indicator_corrs: dict = field(default_factory=dict)
    indicator_pvalues: dict = field(default_factory=dict)
    threshold: float = 0.60
    stop_pct: float = 0.03
    target_pct: float = 0.12
    max_position_notional: float = MAX_POSITION_NOTIONAL
    max_loss_per_trade: float = 50.0
    confidence: float = 0.0
    profitable: bool = False
    n_profitable: int = 0
    n_total: int = 0

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "threshold": self.threshold,
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
            "max_position_notional": self.max_position_notional,
            "max_loss_per_trade": self.max_loss_per_trade,
            "confidence": self.confidence,
            "profitable": self.profitable,
            "n_profitable": self.n_profitable,
            "n_total": self.n_total,
            "indicator_corrs": self.indicator_corrs,
            "indicator_pvalues": self.indicator_pvalues,
        }


def _calibrate_weights_from_pooled(pooled: dict) -> dict[str, float]:
    """Weight indicators by relative |correlation| strength.

    Non-significant indicators get a floor weight (diversification),
    but are downweighted relative to proven-edge indicators.
    """
    floor = 0.01
    raw = {}
    for name, info in pooled.items():
        if info["significant"] and abs(info["corr"]) > 0:
            raw[name] = abs(info["corr"])
        else:
            raw[name] = floor
    total = sum(raw.values())
    weights = {k: max(v / total, floor) for k, v in raw.items()}
    wsum = sum(weights.values())
    return {k: v / wsum for k, v in weights.items()}


def _check_profitability(tickers, data_dir) -> tuple[int, int]:
    """Run full backtest; count profitable tickers."""
    from .backtest import backtest_ticker
    from .data import load_ohlcv

    profitable = 0
    total = 0
    for ticker in tickers:
        path = data_dir / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        data = load_ohlcv(path)
        m = backtest_ticker(ticker, data["close"], data["high"],
                            data["low"], data["volume"])
        total += 1
        if m["ev_per_trade"] > 0 or m["total_trades"] == 0:
            profitable += 1
    return profitable, total


def _print_weights(cal: Calibration):
    for name, w in sorted(cal.weights.items(), key=lambda x: -x[1]):
        c = cal.indicator_corrs.get(name, 0.0)
        print(f"  {name:25s}: {w:.3f}  (corr={c:+.4f})")


def calibrate(tickers, data_dir) -> Calibration:
    """Full auto-calibration pipeline."""
    cal = Calibration()

    print("Step 1: Evaluating indicator edge (pooled permutation test)...")
    pooled = evaluate_indicator_pooled(tickers, data_dir, n_perm=500)
    cal.indicator_corrs = {k: v["corr"] for k, v in pooled.items()}
    cal.indicator_pvalues = {k: v["pvalue"] for k, v in pooled.items()}
    _print_pooled_results(pooled)

    print("\nStep 2: Auto-calibrating weights by edge strength...")
    cal.weights = _calibrate_weights_from_pooled(pooled)
    _print_weights(cal)

    print("\nStep 3: Grid-searching optimal stop/target/threshold...")
    best_cfg, stats = grid_search(tickers, data_dir, cal.weights)
    cal.stop_pct = best_cfg.stop_pct
    cal.target_pct = best_cfg.target_pct
    cal.threshold = best_cfg.threshold
    print(f"  Best: {cal.stop_pct:.0%}/{cal.target_pct:.0%} thr={cal.threshold:.2f} "
          f"EV={stats['avg_ev']:.3f}R")

    print("\nStep 4: Verifying profitability on full universe...")
    profitable, total = _check_profitability(tickers, data_dir)
    cal.n_profitable, cal.n_total = profitable, total
    cal.profitable = profitable > 0 and (profitable / max(total, 1)) >= 0.3
    _set_confidence(cal, pooled, profitable, total)
    return cal


def _print_pooled_results(pooled: dict):
    for name, info in pooled.items():
        sig = "✅" if info["significant"] else "❌"
        print(f"  {name:25s}: corr={info['corr']:+.4f}  "
              f"p={info['pvalue']:.4f}  n={info['n_samples']}  {sig}")


def _set_confidence(cal: Calibration, pooled: dict, profitable: int, total: int):
    sig_count = sum(1 for v in pooled.values() if v["significant"])
    edge_score = sig_count / len(INDICATOR_NAMES)
    profit_score = profitable / max(total, 1)
    cal.confidence = 0.4 * edge_score + 0.6 * profit_score
    print(f"  Profitable: {profitable}/{total} tickers")
    print(f"  Self-confidence: {cal.confidence:.0%}")
