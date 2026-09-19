#!/usr/bin/env python3
"""scripts/ab_brain_replay.py — organ-level A/B replay for gated brain features.

The committed fixtures drift the JULI pipeline through ``hands.py``, which
bypasses the NeuromorphicBrain — so the gated organs (C1 sleep, A1 STDP
learning, D1 adaptive thresholds, G1 MoE routing) can never be measured by
``hanoon_prime.backtest``. This harness instead replays each fixture bar
through the REAL live decision loop (``decide_entry`` → ``check_exit`` →
``on_trade_close``) with all roadmap gates OFF vs each organ flipped, and
emits per-ticker trade metrics + a trials-penalised deflated Sharpe.

The live path itself is NOT touched: gates are flipped at runtime by
patching module constants that are read lazily at call time (or the module
alias for NEURO_BLEND_ENABLED), and every variant runs a fresh brain with
``persist_memory=False`` (no runtime/ state files written).

Trade accounting nets the production cost model (immune.SLIPPAGE_BPS per
fill + edge.compute_fee_drag round trip amortized over the production
account equity) so the P/L funnel is a money-gate lens, not a frictionless
one. Pass ``--no-costs`` for the frictionless research comparison.

Run:
  python scripts/ab_brain_replay.py --data-dir data/fixtures \
      --tickers ALL --variants OFF,BLEND --output reports/ab.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from hanoon_prime.brain.horizons import params_for
from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.policy.verdict import ENTER
from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.immune import ATR_PERIOD, FEE_RATE, FIXED_FEE, SLIPPAGE_BPS
from hanoon_prime.juli_feed import compute_alpha_from_snap

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("ab_replay")

WINDOW: int = 200  # rolling arrays fed to compute_all_alpha
PRICES_HISTORY: int = 56  # snapshot.prices (>= 20 required by validity gate)
SESSION_BARS: int = 390  # bars between forced sleep-replay cycles (C1)
EVAL_MAX_OPEN: int = 5  # cap open positions per replay (portfolio realism)
ACCOUNT_EQUITY: float = 100_000.0  # production-synced account seed for sizing


@dataclass(frozen=True)
class Variant:
    name: str
    blend: bool = False
    learn: bool = False
    threshold: bool = False
    moe: bool = False
    sleep: bool = False
    calibration: bool = False
    hysteresis: bool = False
    probe: bool = False
    contra: bool = False
    delib: bool = False
    halim: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("OFF"),
    Variant("CALIB", calibration=True),
    Variant("HYST", hysteresis=True),
    Variant("PROBE", probe=True),
    Variant("CONTRA", contra=True),
    Variant("DELIB", delib=True),
    Variant("HALIM", halim=True),
    Variant("BLEND", blend=True),
    Variant("BLEND+LEARN", blend=True, learn=True),
    Variant("BLEND+THRESH", blend=True, threshold=True),
    Variant("BLEND+MOE", blend=True, moe=True),
    Variant("BLEND+SLEEP", blend=True, sleep=True),
    Variant(
        "ALL-ON",
        blend=True,
        learn=True,
        threshold=True,
        moe=True,
        sleep=True,
        calibration=True,
        hysteresis=True,
        probe=True,
        contra=True,
        delib=True,
        halim=True,
    ),
)

# Per-ticker hold caps in bars, mirroring the per-horizon stale_minutes window
DEADLINE = {
    h: max(10, int(params_for(h).stale_minutes))
    for h in ("scalp", "multihour", "swing", "multiday", "multiweek", "longterm")
}


@dataclass
class Position:
    ticker: str
    entry_price: float
    direction: int
    bar: int
    atr: float
    horizon: str
    score: float = 0.0


@dataclass
class Trade:
    ticker: str
    direction: int
    r: float
    won: bool
    entry_bar: int
    exit_bar: int
    horizon: str
    reason: str
    score: float = 0.0
    atr_pct: float = 0.0


@dataclass
class RunResult:
    variant: str
    tickers: dict[str, list[Trade]] = field(default_factory=dict)


@dataclass
class Slice:
    """Walk-forward replay bounds (in-sample fraction / out-of-sample tail)."""

    fraction: float = 1.0
    oos_last: float = 0.0

    def bounds(self, n: int, window: int) -> tuple[int, int]:
        start, end = window, n
        if self.fraction < 1.0:
            end = window + int((n - window) * self.fraction)
        if self.oos_last > 0.0:
            start = window + int((n - window) * (1.0 - self.oos_last))
        return start, end


def _apply_variant(v: Variant) -> dict[str, bool]:
    """Patch the roadmap gates (lazy readers + the module-imported aliases)."""
    import hanoon_prime.brain.consolidation as cons
    import hanoon_prime.brain.exit_ladder as xl
    import hanoon_prime.brain.orchestrator as orb
    import hanoon_prime.brain.probe_recovery as pr
    import hanoon_prime.brain.realized_ev as rev
    import hanoon_prime.contrarian as ctr
    import hanoon_prime.immune as imm
    import hanoon_prime.inspection.pillar_evidence as pe

    saved = {
        "blend": orb.NEURO_BLEND_ENABLED,
        "learn": imm.NEURO_LEARN_ENABLED,
        "threshold": imm.NEURO_ADAPTIVE_THRESHOLD_ENABLED,
        "moe": imm.NEURO_MOE_GATE_ENABLED,
        "calibration": rev.CALIBRATION_NUDGE_ENABLED,
        "hysteresis": xl.HYSTERESIS_EXIT_ENABLED,
        "probe": pr.PROBE_RECOVERY_ENABLED,
        "contra": ctr.CONTRARIAN_MODE_ENABLED,
        "delib": orb.DELIBERATION_TRACE_ENABLED,
        "halim_pe": pe.HALIM_EVIDENCE_LEARNING,
        "halim_cons": cons.HALIM_EVIDENCE_LEARNING,
        "entry_regime": orb.ENTRY_REGIME_GATE,
        "entry_cost": orb.ENTRY_COST_AVERSE_GATE,
    }
    orb.NEURO_BLEND_ENABLED = v.blend
    orb.DELIBERATION_TRACE_ENABLED = v.delib
    imm.NEURO_LEARN_ENABLED = v.learn
    imm.NEURO_ADAPTIVE_THRESHOLD_ENABLED = v.threshold
    imm.NEURO_MOE_GATE_ENABLED = v.moe
    imm.CALIBRATION_NUDGE_ENABLED = v.calibration
    imm.HYSTERESIS_EXIT_ENABLED = v.hysteresis
    imm.PROBE_RECOVERY_ENABLED = v.probe
    imm.CONTRARIAN_MODE_ENABLED = v.contra
    imm.HALIM_EVIDENCE_LEARNING = v.halim
    rev.CALIBRATION_NUDGE_ENABLED = v.calibration
    xl.HYSTERESIS_EXIT_ENABLED = v.hysteresis
    pr.PROBE_RECOVERY_ENABLED = v.probe
    ctr.CONTRARIAN_MODE_ENABLED = v.contra
    pe.HALIM_EVIDENCE_LEARNING = v.halim
    cons.HALIM_EVIDENCE_LEARNING = v.halim
    orb.ENTRY_REGIME_GATE = ENTRY_GATE in ("regime", "both")
    orb.ENTRY_COST_AVERSE_GATE = ENTRY_GATE in ("cost", "both")
    return saved


def _restore_variant(saved: dict[str, bool]) -> None:
    """Restore the gates so an in-process consumer sees production defaults."""
    import hanoon_prime.brain.consolidation as cons
    import hanoon_prime.brain.exit_ladder as xl
    import hanoon_prime.brain.orchestrator as orb
    import hanoon_prime.brain.probe_recovery as pr
    import hanoon_prime.brain.realized_ev as rev
    import hanoon_prime.contrarian as ctr
    import hanoon_prime.immune as imm
    import hanoon_prime.inspection.pillar_evidence as pe

    orb.NEURO_BLEND_ENABLED = saved["blend"]
    orb.DELIBERATION_TRACE_ENABLED = saved["delib"]
    imm.NEURO_LEARN_ENABLED = saved["learn"]
    imm.NEURO_ADAPTIVE_THRESHOLD_ENABLED = saved["threshold"]
    imm.NEURO_MOE_GATE_ENABLED = saved["moe"]
    imm.CALIBRATION_NUDGE_ENABLED = saved["calibration"]
    imm.HYSTERESIS_EXIT_ENABLED = saved["hysteresis"]
    imm.PROBE_RECOVERY_ENABLED = saved["probe"]
    imm.CONTRARIAN_MODE_ENABLED = saved["contra"]
    imm.HALIM_EVIDENCE_LEARNING = saved["halim_pe"]
    rev.CALIBRATION_NUDGE_ENABLED = saved["calibration"]
    xl.HYSTERESIS_EXIT_ENABLED = saved["hysteresis"]
    pr.PROBE_RECOVERY_ENABLED = saved["probe"]
    ctr.CONTRARIAN_MODE_ENABLED = saved["contra"]
    pe.HALIM_EVIDENCE_LEARNING = saved["halim_pe"]
    cons.HALIM_EVIDENCE_LEARNING = saved["halim_cons"]
    orb.ENTRY_REGIME_GATE = saved["entry_regime"]
    orb.ENTRY_COST_AVERSE_GATE = saved["entry_cost"]


@dataclass
class SignalResult:
    variant: str
    signals: dict[str, list[dict[str, float]]] = field(default_factory=dict)


def replay_signals(
    fixtures: dict[str, dict[str, Any]], variant: Variant
) -> SignalResult:
    """Signal-channel A/B wrapper: apply the gate, run, always restore."""
    saved = _apply_variant(variant)
    try:
        return _replay_signals_impl(fixtures, variant)
    finally:
        _restore_variant(saved)


def _replay_signals_impl(
    fixtures: dict[str, dict[str, Any]], variant: Variant
) -> SignalResult:
    """Signal-channel A/B: the pre-sizing decision on every bar.

    The sizing funnel (~15 trades per variant over 22 tickers) gives no
    statistical resolution, so this mode measures the organ's actual
    computational effect on the brain's score/side across EVERY bar
    (~37k samples per variant) — the question the promotion decision
    turns on: does the organ move the signal at all?
    """
    brain = NeuromorphicBrain(persist_memory=False)
    res = SignalResult(variant=variant.name)
    for ticker, bars in fixtures.items():
        close = np.asarray(bars["close"], dtype=float)
        n = int(len(close))
        if n <= WINDOW + 60:
            continue
        sigs: list[dict[str, float]] = []
        for i in range(WINDOW, n):
            if variant.sleep and i % SESSION_BARS == 0:
                try:
                    brain._sleep_engine.run_cycle()
                except Exception:
                    pass
            snap = build_snap(bars, i)
            if snap is None:
                continue
            alpha = compute_alpha_from_snap(snap)
            if not alpha:
                continue
            ctx = brain._score_pipeline(ticker, alpha, 1.0, 0.0, 0.0)
            sigs.append(
                {
                    "score": float(ctx["stabilized"]),
                    "side": float(ctx["final_dir"]),
                    "neuro": float(ctx.get("neuro_score", 0.0)),
                }
            )
        res.signals[ticker] = sigs
    return res


def summarize_signal(off: SignalResult, variant: SignalResult) -> dict[str, Any]:
    """Delta of one organ variant vs OFF across the same bar deck."""
    off_scores = [s["score"] for tl in off.signals.values() for s in tl]
    v_scores = [s["score"] for tl in variant.signals.values() for s in tl]
    off_side = [s["side"] for tl in off.signals.values() for s in tl]
    v_side = [s["side"] for tl in variant.signals.values() for s in tl]
    n = min(len(off_scores), len(v_scores))
    if n == 0:
        return {
            "variant": variant.variant,
            "n": 0,
            "mean_delta": 0.0,
            "side_flip": 0.0,
            "corr": 0.0,
            "mean_abs_neuro": 0.0,
        }
    mean_delta = float(
        np.mean(np.abs(np.asarray(v_scores[:n]) - np.asarray(off_scores[:n])))
    )
    flips = float(
        np.mean(
            [
                1.0 if a * b < 0 and b != 0 else 0.0
                for a, b in zip(off_side[:n], v_side[:n])
            ]
        )
    )
    corr = float(np.corrcoef(off_scores[:n], v_scores[:n])[0, 1]) if n > 2 else 0.0
    neuro = [s["neuro"] for tl in variant.signals.values() for s in tl]
    return {
        "variant": variant.variant,
        "n": n,
        "mean_abs_score_delta": float(mean_delta),
        "side_flip_rate": float(flips),
        "score_corr": float(corr),
        "mean_abs_neuro": float(np.mean(np.abs(neuro))) if neuro else 0.0,
    }


def build_snap(bars: dict[str, Any], i: int) -> dict[str, Any] | None:
    """Slice a rolling snapshot at bar ``i`` (None when warmup too short)."""
    close = np.asarray(bars["close"], dtype=float)
    hi = np.asarray(bars["high"], dtype=float)
    lo = np.asarray(bars["low"], dtype=float)
    vol = np.asarray(bars["volume"], dtype=float)
    if i < WINDOW:
        return None
    c = close[i - WINDOW + 1 : i + 1]
    h = hi[i - WINDOW + 1 : i + 1]
    l = lo[i - WINDOW + 1 : i + 1]
    v = vol[i - WINDOW + 1 : i + 1]
    rng = np.maximum(h - l, 1e-12)
    buy = v * np.clip((c - l) / rng, 0.0, 1.0)
    sell = np.maximum(v - buy, 0.0)
    total = np.maximum(v, 1e-12)
    bid = float(np.sum(buy[-10:]) / 10.0 / np.mean(total[-10:]))
    ask = float(np.sum(sell[-10:]) / 10.0 / np.mean(total[-10:]))
    n = int(len(c))
    return {
        "last": float(c[-1]),
        "bid": float(c[-1]),
        "ask": float(c[-1]),
        "mid": float(c[-1]),
        "atr": float(
            np.mean(
                np.maximum(
                    h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
                )[-ATR_PERIOD:]
            )
        ),
        "prices": c[-PRICES_HISTORY:].tolist(),
        "close_arr": c.tolist(),
        "high_arr": h.tolist(),
        "low_arr": l.tolist(),
        "volume_arr": v.tolist(),
        "buy_volume_arr": buy.tolist(),
        "bid_sizes_arr": [bid] * n,
        "ask_sizes_arr": [ask] * n,
    }


def _adverse_fill(price: float, direction: int) -> float:
    """Adverse-slippage fill, mirroring production hands._adverse_fill:
    buyers pay up, sellers take less."""
    return price * (1.0 + SLIPPAGE_BPS / 10000.0 * direction)


def _round_trip_fee_fraction() -> float:
    """Round-trip fee drag as a fraction of notional, amortized over the
    production account equity (edge.compute_fee_drag / equity)."""
    notional = ACCOUNT_EQUITY
    if notional <= 0:
        return 0.0
    return 2.0 * (FIXED_FEE + FEE_RATE * notional) / notional


COSTS: bool = True  # harness toggle: net production slippage/fees in the P/L funnel
ENTRY_GATE: str = "none"  # harness toggle: cost/regime entry strategy gates
# Cohort-analysis trace: every closed trade, flushed to --trace FILE at exit.
TRACE_RECORDS: list[dict[str, Any]] = []


def _close(
    brain: NeuromorphicBrain,
    pos: Position,
    price: float,
    bar: int,
    reason: str,
) -> Trade:
    exit_price = float(price)
    entry_ref = pos.entry_price
    exit_ref = exit_price
    if COSTS:
        entry_ref = _adverse_fill(pos.entry_price, pos.direction)
        exit_ref = _adverse_fill(exit_price, -pos.direction)
    pnl_pct = (exit_ref / entry_ref - 1.0) * pos.direction
    if COSTS:
        pnl_pct -= _round_trip_fee_fraction()
    won = pnl_pct > 0.0
    r = _r_mult(pos, exit_ref, entry_ref)
    brain.on_trade_close(pos.ticker, won, pnl_pct, pos.direction)
    brain.note_exit(pos.ticker)
    TRACE_RECORDS.append(
        {
            "ticker": pos.ticker,
            "direction": pos.direction,
            "horizon": pos.horizon,
            "entry_bar": pos.bar,
            "exit_bar": bar,
            "reason": reason,
            "r": float(r),
            "won": bool(won),
            "score": float(pos.score),
            "atr_pct": float(pos.atr / pos.entry_price),
        }
    )
    return Trade(
        ticker=pos.ticker,
        direction=pos.direction,
        r=float(r),
        won=bool(won),
        entry_bar=pos.bar,
        exit_bar=bar,
        horizon=pos.horizon,
        reason=reason,
        score=float(pos.score),
        atr_pct=float(pos.atr / pos.entry_price),
    )


def _r_mult(pos: Position, exit_ref: float, entry_ref: float) -> float:
    pnl_pct = (exit_ref / entry_ref - 1.0) * pos.direction
    atr_pct = max(pos.atr / entry_ref, 1e-6)
    return pnl_pct / atr_pct


def replay_variant(
    fixtures: dict[str, dict[str, Any]],
    variant: Variant,
    both: bool = False,
    lowbar: float | None = None,
    slice_: Slice | None = None,
) -> RunResult:
    """Funnel-path wrapper: apply the gate, run, always restore."""
    saved = _apply_variant(variant)
    try:
        return _replay_variant_impl(
            fixtures, variant, both=both, lowbar=lowbar, slice_=slice_
        )
    finally:
        _restore_variant(saved)


def _replay_variant_impl(
    fixtures: dict[str, dict[str, Any]],
    variant: Variant,
    both: bool = False,
    lowbar: float | None = None,
    slice_: Slice | None = None,
) -> RunResult:
    """Funnel-path body: the live loop over every fixture bar.

    ``both=True`` is the RESEARCH direction mode (brain.trading_policy
    direction_mode="both"): the live default is long_only, which on this
    SHORT-dominated fixture mix admits ~zero entries — no funnel, no A/B
    power. Both-direction is a labelled research lens only; it never
    reflects the live config.

    ``lowbar`` freezes the entry threshold at a fixed value (research):
    the live Dynamics self-calibrates to the score distribution's ~90th
    percentile, which admits ~1 trade per 1900 bars — too thin an A/B
    funnel. Fixed thresholds trade conviction for power. Label only.

    Trade accounting nets production slippage+fees via the module ``COSTS``
    (the money-gate lens); ``--no-costs`` switches it off for the
    frictionless research comparison.
    """
    brain = NeuromorphicBrain(persist_memory=False)
    if both:
        brain.trading_policy.direction_mode = "both"
    if lowbar is not None:
        brain.dynamics._threshold = float(lowbar)
        brain.dynamics._update_quintile_threshold = lambda: None
    # Mirror production post-sync state: RiskEngine sizes from account equity;
    # without it every admission lands on sizing=0 -> HOLD not_sized.
    brain.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "equity_synced": True,
            "equity": ACCOUNT_EQUITY,
        },
        account_feed={
            "equity": ACCOUNT_EQUITY,
            "daily_pnl": 0.0,
            "positions": {},
        },
    )
    result = RunResult(variant=variant.name)
    open_pos: dict[str, Position] = {}

    for ticker, bars in fixtures.items():
        close = np.asarray(bars["close"], dtype=float)
        n = int(len(close))
        if n <= WINDOW + 60:
            log.warning("%s too short, skipping", ticker)
            continue
        trades: list[Trade] = []
        start, end = (slice_ or Slice()).bounds(n, WINDOW)
        for i in range(start, end):
            if variant.sleep and i % SESSION_BARS == 0:
                try:
                    brain._sleep_engine.run_cycle()
                except Exception:
                    pass
            brain.begin_entry_cycle()
            snap = build_snap(bars, i)
            if snap is None:
                continue
            held_names = set(open_pos)
            pos = open_pos.get(ticker)
            if pos is not None and i > pos.bar:
                pnl_d = (close[i] - pos.entry_price) * pos.direction
                sig = brain.check_exit(
                    ticker,
                    float(close[i]),
                    ib_pnl=float(pnl_d),
                    direction=pos.direction,
                )
                if sig.should_exit:
                    trades.append(_close(brain, pos, close[i], i, sig.reason))
                    open_pos.pop(ticker, None)
                elif i - pos.bar >= DEADLINE.get(pos.horizon, 120):
                    trades.append(_close(brain, pos, close[i], i, "horizon"))
                    open_pos.pop(ticker, None)
            if open_pos.get(ticker) is None and len(open_pos) < EVAL_MAX_OPEN:
                v = brain.decide_entry(ticker, snap, held_names, "rth")
                if v.action == ENTER and v.direction != 0:
                    brain.note_entry(ticker)
                    atr = float(snap["atr"])
                    open_pos[ticker] = Position(
                        ticker=ticker,
                        entry_price=float(close[i]),
                        direction=int(v.direction),
                        bar=i,
                        atr=max(atr, 1e-6),
                        horizon=v.horizon,
                        score=float(v.score),
                    )
        for pos in list(open_pos.values()):
            if pos.ticker == ticker and pos.bar < end:
                trades.append(_close(brain, pos, close[end - 1], end - 1, "flush"))
                open_pos.pop(ticker, None)
        result.tickers[ticker] = trades
    return result


def _deflated_sharpe(r: list[float], trials: int) -> dict[str, float]:
    """Bailey & López de Prado closed-form deflation (counts it with honesty).

    ``trials`` = number of variants compared in this A/B — the multiplicity
    penalty that keeps a lucky best-of-K variant from being an edge.
    """
    if len(r) < 4:
        return {"n": len(r), "sr": 0.0, "sr_expected_max": 0.0, "deflated": 0.0}
    a = np.asarray(r, dtype=float)
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if sd <= 1e-12:
        return {"n": len(r), "sr": 0.0, "sr_expected_max": 0.0, "deflated": 0.0}
    sr = mu / sd
    n = float(len(a))
    skew = float(stats.skew(a))
    kurt = float(stats.kurtosis(a, fisher=True))
    gamma = 0.5772156649
    v_est = (1.0 + 0.5 * sr**2 - skew * sr + 0.25 * kurt * sr**2) / n
    v_est = max(v_est, 1e-12)
    trials_f = max(2, int(trials))
    z_max = (1.0 - gamma) * stats.norm.ppf(
        1.0 - 1.0 / trials_f
    ) + gamma * stats.norm.ppf(1.0 - 1.0 / (trials_f * np.e))
    sr_max = float(np.sqrt(v_est)) * z_max
    denom = float(np.sqrt(max((1.0 - skew * sr + 0.25 * kurt * sr**2), 1e-12)))
    deflated = (sr - sr_max) / denom if denom > 0 else 0.0
    return {"n": len(r), "sr": sr, "sr_expected_max": sr_max, "deflated": deflated}


def summarize(result: RunResult, trials: int) -> dict[str, Any]:
    """Per-ticker EV/win-rate + pooled deflated Sharpe for one variant."""
    all_r: list[float] = []
    tickers: dict[str, dict[str, Any]] = {}
    for ticker, trades in result.tickers.items():
        if not trades:
            tickers[ticker] = {"trades": 0, "ev_per_trade": 0.0, "win_rate": 0.0}
            continue
        r = [t.r for t in trades]
        wins = [x for x in r if x > 0]
        rr = (
            float(np.mean(wins)) / abs(float(np.mean([x for x in r if x <= 0])))
            if wins and any(x <= 0 for x in r)
            else 0.0
        )
        t_stats = _deflated_sharpe(r, trials)
        tickers[ticker] = {
            "trades": len(trades),
            "ev_per_trade": float(np.mean(r)),
            "win_rate": float(np.mean([1.0 if t.won else 0.0 for t in trades])),
            "realized_rr": float(rr),
            "sharpe": t_stats["sr"],
            "deflated": t_stats["deflated"],
        }
        all_r.extend(r)
    pooled = _deflated_sharpe(all_r, trials)
    profitable = sum(1 for m in tickers.values() if m["ev_per_trade"] > 0)
    return {
        "variant": result.variant,
        "total_trades": len(all_r),
        "profitable_tickers": profitable,
        "with_trades": sum(1 for m in tickers.values() if m["trades"] > 0),
        "ev_per_trade": float(np.mean(all_r)) if all_r else 0.0,
        "pooled_sr": pooled["sr"],
        "deflated_sr": pooled["deflated"],
        "sr_expected_max": pooled["sr_expected_max"],
        "tickers": tickers,
    }


def load_fixtures(data_dir: Path, tickers: list[str]) -> dict[str, dict[str, Any]]:
    fixtures: dict[str, dict[str, Any]] = {}
    for name in tickers:
        path = data_dir / f"{name}_1min.csv"
        if not path.exists():
            continue
        try:
            fixtures[name] = load_ohlcv(path)
        except Exception as e:
            log.warning("%s: %s", name, e)
    return fixtures


def hermetic_state() -> None:
    """Redirect all brain persistence to a per-process sandbox.

    The funnel drives the REAL live decision loop, and the live brain binds
    every state file (juli_state, horizons, learning_config store,
    attractor memory, EVAL log) to the production ``runtime/`` dir through
    ``config.STATE_DIR``-derived module constants. Replaying through those
    paths would race the running bot over its own state files — so we
    rebind every bound constant to a sandbox dir (AB_HERMETIC_DIR, else a
    /tmp default) and point HANOO_MEMORY_FILE at it. Call-time readers
    (adaptive_thresholds, bridge._memory_path, memory) already follow
    ``config.STATE_DIR`` / the env var, so rebinding config is enough for
    them; only the at-import ``STATE_DIR / <name>`` constants need a
    per-module rebind here.
    """
    base = Path(
        os.environ.get("AB_HERMETIC_DIR") or f"/tmp/ab_replay_{os.getpid()}"
    ).resolve()
    runtime = base / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    # The live brain's persistence layer honours a HANOO_*_FILE env family
    # (designed for exactly this hermetic sandbox use). Point every member at
    # the sandbox so the funnel never touches production runtime/ files.
    for key, name in (
        ("HANOO_MEMORY_FILE", "juli_state.json"),
        ("HANOO_REALIZED_FILE", "juli_realized.json"),
        ("HANOO_ALLOSTASIS_FILE", "juli_allostasis.json"),
        ("HANOO_EXTINCTION_FILE", "juli_extinction.json"),
        ("HANOO_METACOG_FILE", "juli_metacog.json"),
        ("HANOO_REGIME_FILE", "juli_regime_weights.json"),
        ("HANOO_RPE_FILE", "juli_rpe.json"),
    ):
        os.environ[key] = str(runtime / name)

    from hanoon_prime.brain import config

    config.STATE_DIR = runtime
    config.JULI_STATE_FILE = runtime / "juli_state.json"

    from hanoon_prime.brain import halim_evidence, horizons, learning_config

    horizons._STATE_PATH = runtime / "horizons.json"
    halim_evidence.EVAL_LOG_PATH = base / "logs" / "hanoon_prime.log"

    from hanoon_prime.reflection import buffer as _trade_buffer

    _trade_buffer._BUFFER_PATH = base / "models" / "buffer" / "trades.json"

    for attr in (
        "META_FILE",
        "BANDIT_FILE",
        "REGIME_FILE",
        "RPE_FILE",
        "ALLOS_FILE",
        "EXTINCT_FILE",
        "METACOG_FILE",
        "STRATEGY_FILE",
        "STRATEGY_BANDIT_FILE",
        "SHADOW_BOOK_FILE",
    ):
        setattr(learning_config, attr, runtime / getattr(learning_config, attr).name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--variants", default="OFF")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--both",
        action="store_true",
        help="RESEARCH: direction_mode='both' (live default is long_only; "
        "under it these fixtures admit ~zero brain entries)",
    )
    parser.add_argument(
        "--lowbar",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="RESEARCH: freeze the entry threshold at THRESHOLD (live self-"
        "calibration admits ~1 trade per 1900 bars); gives the A/B funnel "
        "enough trades to be discriminative",
    )
    parser.add_argument(
        "--signal",
        action="store_true",
        help="Run the SIGNAL-channel A/B: pre-sizing score/side per bar for "
        "every variant (well-powered — ~37k samples/variant) instead of the "
        "P/L funnel (only ~15 trades/variant on these fixtures)",
    )
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="frictionless research comparison (net costs are the money-gate lens)",
    )
    parser.add_argument(
        "--entry-gate",
        choices=("none", "regime", "cost", "both"),
        default="none",
        help="Apply the money-gate strategy layers to every variant run: "
        "'regime' only enters non-trending/non-volatile local regimes, "
        "'cost' requires ATR%% to clear ENTRY_COST_CAPTURE_MULTIPLE x the "
        "round-trip cost fraction, 'both' applies both. Uses the SAME "
        "production decide_entry — strategy-gate verdicts (cost_floor / "
        "regime_adverse) are produced by live-identical code.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        metavar="F",
        help="WALK-FORWARD: replay only the first F of each fixture "
        "(in-sample selection window); a confirm run then re-uses the frozen "
        "params on --oos-last (out-of-sample holdout).",
    )
    parser.add_argument(
        "--oos-last",
        type=float,
        default=0.0,
        metavar="F",
        help="WALK-FORWARD: replay only the LAST F of each fixture — the "
        "out-of-sample holdout for confirming a screened strategy.",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE",
        default=None,
        help="Cohort-analysis trace: dump every closed trade "
        "(ticker/direction/horizon/score/atr_pct/r/reason) as JSON.",
    )
    args = parser.parse_args()
    hermetic_state()
    global COSTS, ENTRY_GATE
    COSTS = not args.no_costs
    ENTRY_GATE = args.entry_gate

    data_dir = Path(args.data_dir)
    ticker_arg = (
        [p.stem.replace("_1min", "") for p in sorted(data_dir.glob("*_1min.csv"))]
        if args.tickers.upper() == "ALL"
        else [t.strip() for t in args.tickers.split(",")]
    )
    selected = {
        v.name
        for v in VARIANTS
        if v.name in args.variants.split(",") or args.variants.upper() == "ALL"
    }
    fixtures = load_fixtures(data_dir, ticker_arg)
    if not fixtures:
        log.error("No fixtures loaded from %s", data_dir)
        return 1
    summaries = []
    if args.signal:
        off = None
        for variant in VARIANTS:
            if variant.name not in selected:
                continue
            res = replay_signals(fixtures, variant)
            if variant.name == "OFF":
                off = res
                continue
            sig = summarize_signal(off, res)
            summaries.append(sig)
            print(
                f"{sig['variant']:>14}: n={sig['n']:6d} "
                f"mean|Δscore|={sig['mean_abs_score_delta']:+.5f} "
                f"side_flip={sig['side_flip_rate']:.4f} "
                f"corr={sig['score_corr']:+.3f} "
                f"mean|neuro|={sig['mean_abs_neuro']:.5f}"
            )
    else:
        for variant in VARIANTS:
            if variant.name not in selected:
                continue
            res = replay_variant(
                fixtures,
                variant,
                both=args.both,
                lowbar=args.lowbar,
                slice_=Slice(fraction=args.fraction, oos_last=args.oos_last),
            )
            s = summarize(res, trials=len(VARIANTS))
            s["lens"] = {
                "costs": COSTS,
                "entry_gate": ENTRY_GATE,
                "both": args.both,
                "lowbar": args.lowbar,
                "slice": {
                    "fraction": args.fraction,
                    "oos_last": args.oos_last,
                },
            }
            summaries.append(s)
            print(
                f"{variant.name:>14}: trades={s['total_trades']:5d} "
                f"prof={s['profitable_tickers']:2d}/{s['with_trades']:2d} "
                f"EV/R={s['ev_per_trade']:+.3f} pooled_SR={s['pooled_sr']:+.3f} "
                f"deflated_SR={s['deflated_sr']:+.3f}"
            )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2))
    if args.trace and TRACE_RECORDS:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(TRACE_RECORDS, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
