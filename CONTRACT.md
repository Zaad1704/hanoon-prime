# HANOON PRIME — Architectural Contract

> **Single invariant: no code merges without backtest proof.**
> If a change cannot demonstrate positive expectancy on out-of-sample
> historical data, it does not belong in this codebase.

## Architecture: IB → JULI → Entry/Hold/Exit + Learning

```
┌─────────┐     ┌──────────────────┐     ┌────────────────────┐
│  IB     │  → │  JULI (brain)    │  →  │  Decisions + Learn  │
│  1min   │     │  alpha → score   │     │  • Entry verdict    │
│  OHLCV  │     │  → EV → think    │     │  • Exit verdict      │
│  + fills│     │  → exit logic    │     │  • Weight update     │
└─────────┘     └──────────────────┘     └────────────────────┘
```

- **IB is the single source of truth** for market data and fills. There is no
  local data store for prices — only a thin cache for latency smoothing.
- **JULI is the sole decision-maker.** The brain receives alpha from IB data,
  produces a score, computes EV, and emits ENTER/HOLD/EXIT. No external system
  overrides the verdict — only **hard safety nets** (stops, daily loss limits,
  position caps) can prevent an entry, and those are enforced as exceptions,
  not score modifiers.
- **Learning is ONE system only** until proven safe: online weight adaptation
  via logistic-regression gradient on realized trade outcomes. No separate
  neuromorphic, sentiment, emotion, or gate-advisor learning subsystems exist.

## Rules (enforced in CI)

| Rule | Check | Fail if |
|------|-------|---------|
| R1 | `R1_single_path` test | More than one function calls the thinker's verdict |
| R2 | `R2_profit_gate` CI job | Backtest expectancy < 0 on any ticker |
| R3 | `R3_complexity` pre-commit | Any function > 40 lines or nesting > 3 |
| R4 | `R4_indicator_edge` test | Any indicator shows no positive correlation with next-bar return |
| R5 | `R5_no_inversion` test | SCORE_INVERT is True or win_prob > PRIOR_TOP_MAX |
| R6 | `R6_safety_nets` test | Any safety net can be bypassed via config flag |
| R7 | `R7_immutable_journal` test | Journal entries can be deleted or updated |
| R8 | `R8_one_learning_system` test | More than one file implements weight adaptation |

## Complexity Budget

| Component | Max Lines |
|-----------|-----------|
| `hanoon/alpha.py` | 200 |
| `hanoon/scoring.py` | 200 |
| `hanoon/thinker.py` | 200 |
| `hanoon/brain.py` | 200 (orchestrator only — no logic) |
| Any single function | 40 lines |
| Any single function nesting | 3 levels |

## Indicators (exactly 5, no more)

1. **VPIN** — volume-synchronized probability of informed trading
2. **Orderbook imbalance** — bid/ask size pressure (from IB depth of market)
3. **Institutional flow** — volume-spike proxy for large-trader activity
4. **Momentum** — price change over 5-period lookback
5. **VWAP deviation** — distance from volume-weighted average price

Every indicator must pass `R4_indicator_edge` before it is admitted. Non-performing
indicators are **deleted**, not dampened.
