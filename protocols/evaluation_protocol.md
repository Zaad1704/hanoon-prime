# Hanoon Prime — Evaluation Protocol (Phase 4, PRE-LOCKED)

Status: **LOCKED as of commit `P4_PROTOCOL_COMMIT`**. Do not edit in place.
A protocol change is a new version with a new commit + new release criteria;
an edit to this file invalidates any verdict derived from it (the paper
harness records the protocol hash in its output for auditability).

## 1. Data & instrument set

- Source: recorded live-feed 1-min OHLCV CSV per ticker (same format as
  `data/fixtures/`; the live feed appends sessions to `data/market_data/`).
- Universe: every ticker present in the live-feed directory when the paper
  session starts. No exclusions after the fact.
- Only bars with `datetime` in `[session_start, session_end)` count toward a
  session. A session is one trading day's bars.

## 2. Simulation config (locked)

- Window/edge lookback: `EDGE_LOOKBACK` = **50 bars** (never changed).
- Baseline brain: `Hippocampus()` with static `INDICATOR_WEIGHTS` (learning
  OFF in the backtest/walk-forward path — matches how WFA already runs).
- Fills: existing slippage/adverse-fill model stays ON (`_adverse_fill`).
- Risk rails: all immune caps/safety nets ON (notional cap, loss cap,
  consecutive-loss pause, daily-loss halt, latched kill switch). The paper
  harness reads the same `immune` constants live — it never mocks them.
- **No runtime module patching** of `hands`/`cortex` constants in the paper
  harness. Ablation is a *finding* step; the paper evaluates the shipped,
  unmodified organs only.

## 3. Statistical protocol (pre-locked, from Phase 2)

- WFA folds per ticker: **6 contiguous OOS folds** (`DEFAULT_FOLDS`).
- Min completed OOS trades per ticker for admissibility: **30** (`MIN_TRADES`).
- Universe PASS requires: deflated OOS Sharpe **> 0.05** and PBO **< 0.05**
  over all admissible tickers, same definitions as `wfa.py`/`wfa_report.py`.
- A universe with zero admissible tickers = **INSUFFICIENT = FAIL** (never
  passes on missing data).

## 4. Paper release criteria (pre-locked — must ALL hold for Phase 5)

| # | Criterion | Locked value |
|---|-----------|--------------|
| P1 | Minimum completed paper trades | **≥ 30** (same as Phase-2 MIN_TRADES) |
| P2 | Minimum trading sessions observed | **≥ 5** |
| P3 | Session kill-switch observation | **≤ 0** sessions where a latched `KILL_DAILY_LOSS_LIMIT` ($500) breach would have fired in paper |
| P4 | Session daily-loss observation | **≤ 1** session at `DAILY_LOSS_LIMIT` ($200) |
| P5 | WFA universe verdict | **PASS** under Section 3 (deflated > 0.05, PBO < 0.05) |
| P6 | No runtime `ValueError`/silent-skip on any ticker | **0 errors** (data present, simulation completed) |

`scripts/paper_run.py` implements exactly these checks and refuses to invent
new ones. Report writes `reports/phase4_paper.json/.md`; exit 0 only when all
P1–P6 hold (a *FAIL* now is the honest, expected outcome on thin data).

## 5. What counts and what doesn't

- Counted: completed trades (filled entry and exit) inside observed sessions.
- Not counted: open positions at session (data) end, warmup bars, trades
  aborted by a safety rail (rail abort is a *safety* event, reported in
  `safety_events`, never silently dropped).
- Sizing: paper notional uses the same static caps as the backtest. P&L is
  reported per-share R and in USD estimate (cap × R).

## 6. Verification

To reproduce any release verdict: rerun `scripts/paper_run.py` on the same
recorded data directory and compare protocol hash + verdict. Hash mismatch →
the run is a *different protocol* and is not comparable.