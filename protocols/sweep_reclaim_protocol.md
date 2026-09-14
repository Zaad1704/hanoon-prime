# Phase 14 — PDH/PDL Sweep-and-Reclaim (Pre-Registered Protocol)

Hash-pinned strategy parameters frozen BEFORE any bench runs. The values
below may not be tuned after results are seen. Any tweak = a NEW protocol
version with a new hash; the old one dies.

## 1. Idea

Liquidity-sweep fade at the prior session's High/Low. When 1-min price
wicks beyond prior-day low (PDL) then reclaims back above, a cluster of
stops was just harvested; the snap-back targets the range midpoint. Mirror
for a sweep above prior-day high (PDH). This is the single structure the
external evidence (SSRN ORB study, RTY sweep-reversal backtests) isolates
as carrying edge — high-RVOL names, morning window, no overnight.

## 2. Universe

All tickers present in the research data dir (RTH 1-min CSV panel). Gate:
no price > $1, no pre-filtering on the bench panel. RVOL is NOT a hard
filter here — it is REPORTED per signal (volume of reclaim bar / 20-bar
median) so the bench can show whether RVOL>1.2 dominates the edge
(pre-registered hypothesis, not a tuned filter).

## 3. Signal rules (no lookahead)

- Levels come from the PRIOR session only: PDL, PDH, session Mid,
  daily ATR(14) over prior 14 sessions. Nothing from the current day is
  used before the current minute exists.
- Entry window: first entry at/after `10:00` ET, last new entry before
  `11:30` ET. No trades before the opening range establishes, none after
  the morning window (evidence: short-window domination).
- Long setup: while no trade open this session, a 1-min bar prints
  `low < PDL` (sweep), THEN a bar closes `close >= PDL` (reclaim) within
  `SWEEP_MAX_BARS` of the sweep bar. Enter at the reclaim close.
  Sweep extreme tracks the minimum low between sweep and reclaim.
- Short setup: mirror at `high > PDH`, reclaim = `close <= PDH`, enter at
  the reclaim close, sweep extreme = maximum high.
- Stale-sweep expiry: if no reclaim within `SWEEP_MAX_BARS`, the sweep
  is dropped (no entry).
- Minimum R:R: trade rejected when
  `(mid - entry)/(entry - stop) < MIN_RR` (long) or the mirrored short
  R:R < `MIN_RR`. Rationale: a swept-deep name with a far stop has no
  structural snap-back math left.
- One trade per ticker per session (max). If entry is taken, the session
  is done regardless of exit reason.

## 4. Exit rules

- Stop: sweep extreme - `STOP_CUSHION_MULT * ATR_prev` (long) /
  + same (short). Structural invalidation = the harvested level failed.
- Target: session Mid = (PDH + PDL) / 2 (the opposing liquidity zone).
- Time exit: flat at `SESSION_CLOSE` (`15:50` ET). No overnight.

## 5. Market context veto (SPY)

Only when a SPY 1-min file exists in the same data dir. Long signals are
skipped unless SPY's close at the entry minute is above SPY's session
open; shorts skipped unless below. This is the "don't fight macro flow"
gate, wired as a hard skip (not a score nudge). Absent SPY data -> veto
off and flagged in the report.

## 6. Frozen parameters

| Param | Value | Meaning |
| --- | --- | --- |
| ENTRY_FROM | 10:00 ET | first allowed entry minute |
| ENTRY_TO | 11:30 ET | last allowed entry minute |
| SESSION_CLOSE | 15:50 ET | force-flat time |
| SWEEP_MAX_BARS | 10 | max bars between sweep and reclaim |
| STOP_CUSHION_MULT | 0.25 | stop cushion as fraction of ATR_prev |
| MIN_RR | 1.0 | min R:R to accept a signal |
| FEE_R | 0.02R | per-trade all-in cost drag in R units |

## 7. Report card (same surface as phases 7-13)

Per ticker: OOS trades, R-expectancy, Sharpe (per-trade basis). Universe:
pooled Sharpe, deflated edge, PBO, verdict PASS/FAIL/INSUFFICIENT via
`wfa.verdicts` (MIN_TRADES=30 floor). Accounting: pnl per trade in R
multiples; net = gross R minus FEE_R. PASS here does NOT arm live capital
— MicroLiveGuard still requires the paper stage. Override flags reported,
never enforced: RVOL decomposition, SPY-free run.