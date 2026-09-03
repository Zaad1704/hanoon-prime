# HANOON PRIME 3.0 — The Procrustean Production Contract

> **Single invariant: no code merges without backtest proof.**
> If a change cannot demonstrate positive expectancy on out-of-sample
> historical data, it does not belong in this codebase.

This is not a style guide. This is a **production contract** — a set of
hard, tooling-enforced constraints that make spaghetti code structurally
impossible. If you can't commit, you can't break the main branch.

---

## Architecture: IB Gateway → JULI Neuromorphic Brain

```
┌────────────┐        ┌─────────────────────────────────────────────┐
│  IB Gateway  │ →   │  JULI BRAIN (single consciousness)          │
│  Live Stream │    │                                             │
│  reqMktData  │    │  ┌───────┐  ┌────────────┐  ┌────────┐     │
│  reqMktDepth │    │  │ EYES  │→ │CEREBELLUM  │→ │CORTEX  │     │
│  reqHistData │    │  │(data) │  │(signals)   │  │(score) │     │
│  reqPnL      │    │  └───────┘  └──────┬──────┘  └────┬───┘     │
│  bracketOrd  │    │                   │                │          │
│  pendingTkr  │    │  ┌────────────────┘    ┌─────────┘          │
│  waitOnUpd  │    │  ▼                        ▼                   │
└────────────┘    │ ┌────────┐   ┌──────────────────┐           │
                  │ │ HANDS  │   │ IMMUNE SYSTEM    │           │
                  │ │(execute│   │ (hard-coded safety)│           │
                  │ │ brackets│   └────────┬─────────┘           │
                  │ │ ATR stops) │         │                    │
                  │ └────────┘               │                    │
                  │                          │                    │
                  │ ┌────────────────────────┘                    │
                  │ ▼                                              │
                  │ ┌──────────────┐    ┌──────────┐             │
                  │ │ HIPPOCAMPUS  │    │ MEMORY   │             │
                  │ │ (learning)   │ →  │(journal) │             │
                  │ └──────────────┘    └──────────┘             │
                  └─────────────────────────────────────────────┘

IB Adapter Layer (4 files):
  ib_compat.py   → Python 3.14 event loop shim for ib_insync
  ib_streamer.py → reqMktData + reqMktDepth + reqHistoricalData → StreamBuffer
  ib_executor.py → bracketOrder() + exits + journal stamping
  ib_adapter.py  → main event loop: pendingTickers → Cortex → bracket
```

### Key Principles

1. **IB is the Single Source of Truth**: JULI does NOTHING locally that IB
   already provides. Positions, P&L, orders, fills — all come from IB.
   The system is like a human trader who views, selects, buys, sells, and
   holds entirely through IB TWS. Journal is a carbon copy of IB state.
2. **Single Consciousness**: JULI is ONE process. IB socket → ring buffer →
   cerebellum → cortex → immune check → hands. Every component shares memory.
3. **Parallel Processing**: JULI has TWO independent screens (LONG/SHORT)
   running simultaneously. Each ticker gets its own cerebellum/cortex instance.
4. **Punishment-Dominant Learning**: JULI is punished 2× harder for losses
   than rewarded for wins. Weight decay (0.999/trade) prevents paralysis.
5. **Hard Safety Nets**: The Immune System is non-overridable.
   IB-reported daily loss → hard stop. 3 consecutive IB fills → pause.
   3 max concurrent positions. $50 max loss per trade.
6. **Limit Orders Only**: Exits use limit orders, not market orders.
   Market fills are random and unpredictable. Limit orders give control.

---

## Rules (Enforced in CI + Pre-commit)

| Rule | Enforced By | Fail if |
|------|------------|---------|
| R1 | `test_R1_*` + `check_verdict_strings.py` | Signal modules produce verdict strings |
| R2 | `check_profit_gate.py` CI job | Backtest EV ≤ 0 on any ticker with trades |
| R3 | `check_complexity.py` | Any function > 40 lines or nesting > 3 |
| R3b | `check_file_length.sh` | Any source file > 200 lines |
| R3c | `mypy --strict` | Type errors anywhere in src/ |
| R4 | `test_R4_*` | Fewer or more than exactly 5 indicators |
| R5 | `test_R5_*` | SCORE_INVERT=True or PRIOR_TOP > 0.65 |
| R6 | `test_R6_*` | Safety nets configurable via env vars |
| R7 | `test_R7_*` | Journal entries can be deleted/updated |
| R8 | `test_R8_*` | Weight adaptation in any file except hippocampus.py |
| R9 | `test_R9_*` + `check_positive_weights.py` | Any INDICATOR_WEIGHTS value < 0 |
| R10 | `test_R10_*` + `check_print.py` | Any `print()` call in src/ (excluding ib_adapter.py) |
| R11 | `test_R11_*` + `check_docstrings.py` | Public function missing docstring |
| R12 | `test_R12_*` | Coverage gate not configured at ≥80% |
| R13 | `test_R13_*` + `check_no_string_dispatch.py` | Verdicts compared as strings (typo risk) |
| R14 | `test_R14_*` | Constants without type annotations |
| R15 | `test_R15_*` + `check_no_swallow.py` | Bare `except:` or `except: pass` |
| R16 | `test_R16_*` + `check_no_todo.py` | TODO/FIXME/HACK/XXX markers in source |
| R17 | `test_R17_*` | mypy strict mode not enabled |
| R18 | `test_R18_*` | Module missing module-level docstring |

---

## Complexity Budget (R3)

| Component | Max Lines | Max Function Lines | Max Nesting |
|-----------|-----------|--------------------|-------------|
| Any source file | 200 | — | — |
| Any function | — | 40 | 3 levels |

## Indicators (exactly 5, no more — R4)

1. **VPIN** — Volume-synchronized probability of informed trading
2. **Order Book Imbalance** — Bid/ask size pressure (from depth)
3. **Institutional Flow** — Buy-volume delta over 10 bars
4. **Momentum** — 14-bar price change normalized by ATR(14)
5. **VWAP Deviation** — Price distance from VWAP normalized by rolling std

Every indicator must pass `R4_indicator_edge` (permutation test, p < 0.05) before
it is admitted. Non-performing indicators are **deleted**, not dampened.

## Current Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| INDICATOR_WEIGHTS | vpin=0.10, OBI=0.15, inst=0.30, momo=0.20, vwap=0.25 | All positive — FAST tickers show momentum persistence |
| ENTRY_THRESHOLD | 0.65 | \|tanh score\| threshold for entry |
| ATR Stop | 2.0× ATR(14) | 2.0× volatility filter |
| ATR Target | 6.0× ATR(14) | 3:1 R:R ratio |
| R:R | 3:1 | 2:1 stop, 6:1 target |
| SCORE_INVERT | False | No band-aid flipping |
| PRIOR_TOP | 0.60 | Max win probability (capped at 0.65) |
| FEE_RATE | 0.0001 (0.01%/leg) | Institutional ECN pricing |
| FIXED_FEE | $0.01/leg | Negligible round-trip |
| KELLY_FRACTION | 0.25 (25%) | Fractional Kelly for capital preservation |
| TIMEOUT_BARS | 999 (disabled) | ATR barriers decide entry/exit |
| Short allowed | True | Dual LONG/SHORT screens |

## Learning System (R8 — Single System Only)

Only `hippocampus.py` implements weight adaptation. The update rule:

```
On WIN (pnl > 0):   w_i += +REWARD_SCALE × LEARNING_RATE × z_i × direction
On LOSS (pnl < 0):  w_i += -PENALTY_SCALE × LEARNING_RATE × z_i × direction
Between trades:     w_i *= 0.999  (geometric decay)
Bounds:             w_i ∈ [-2.0, +2.0]
```

- LEARNING_RATE = 0.02
- REWARD_SCALE = 0.5 (gentle reinforcement)
- PENALTY_SCALE = 2.0 (strong punishment — 4× harder than reward)
- WEIGHT_DECAY = 0.999 per trade
- WEIGHT_MIN = -2.0, WEIGHT_MAX = +2.0

No other file may implement weight adaptation. All learning goes through
`hippocampus.record_trade()`.

## Safety Nets (R6 — Hard Constants)

| Limit | Value | Description |
|-------|-------|-------------|
| MAX_POSITION_NOTIONAL | $5,000 | Max notional per trade |
| MAX_LOSS_PER_TRADE | $50 | Hard cap per individual trade |
| MAX_CONCURRENT_POSITIONS | 3 | Max open positions simultaneously |
| DAILY_LOSS_LIMIT | $200 | Daily loss → hard stop |
| CONSECUTIVE_LOSSES_PAUSE | 3 | 3 consecutive losses → 60-min pause |
| PAUSE_DURATION_MIN | 60 | Pause duration in minutes |
| MAX_SPREAD_BPS | 5.0 | Max bid/ask spread in basis points |

All values are literal constants in `immune.py`. No `os.environ`, no config
files, no CLI flags. The system cannot be weakened at runtime.

## Data Flow Contract

### Live Mode (IB Direct — Primary Path)

```
IB Gateway (reqMktData + reqMktDepth + reqPnL)
    → ib_streamer.StreamBuffer (circular buffer)
    → cerebellum.compute_alpha() → 5 raw indicator values
    → cortex.evaluate() → Thought(direction, score, z_scores)
    → immune check (positions from ib.positions())
    → ib_executor.place_bracket() → IB bracketOrder()
    → IB handles exits via bracket children (stop/target)
    → ib_executor._journal_snapshot() → carbon copy of IB state
    → hippocampus.record_trade() → asymmetric weight adaptation
```

**IB is the source of truth for:**
- Positions: `ib.positions()`
- P&L: `reqPnL()` streams live daily P&L
- Orders: `ib.trades()` shows all pending/filled orders
- Fills: `trade.fill` objects report actual execution prices

**The system does NOT:**
- Maintain local position state that could drift from IB
- Compute P&L independently when IB provides it
- Use market orders (random fills) — only limit orders
- Generate synthetic journal entries — only carbon copies IB state

### Backtest Mode (CSV — Validation Only)

```
CSV (1-min OHLCV) → eyes.load_ohlcv() → numpy arrays
    → cerebellum.compute_alpha() → 5 raw indicator values
    → cortex.evaluate() → Thought(direction, score, z_scores)
    → hands._enter_position() → Position with ATR stop/target
    → hands._process_bar() → bar-by-bar exit logic
    → memory.Journal.append() → immutable hash-chained log
    → hippocampus.record_trade() → asymmetric weight adaptation
```

Backtest mode exists ONLY to validate strategy before going live.
The live system never touches CSV files.

## Excluded Files

- **`ib_adapter.py`** — Live IB connection adapter. Excluded from mypy,
  contract tests, and R10/R13/R15 checks. This is the boundary to the
  external world; it converts IB data into the numpy arrays the brain
  expects. Business logic never lives here.
- **`ib_streamer.py`** — IB streaming data layer (reqMktData, reqMktDepth,
  reqHistoricalData). Excluded from mypy, R10/R11/R13/R15 checks. No business
  logic — only IB Ticker → numpy array extraction.
- **`ib_executor.py`** — IB order execution layer (bracketOrder, exits).
  Excluded from mypy, R10/R11/R13/R15 checks. Only converts verdicts into
  IB orders and stamps the journal.
- **`ib_compat.py`** — ib_insync import compatibility shim (Python 3.14
  event loop fixup). Excluded from all checks. 29 lines of boilerplate.
