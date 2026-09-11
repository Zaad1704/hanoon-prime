# Hanoon Prime — Recommendations (Phase 6)

Date: 2026-09-12
Scope: state what is proven, what isn't, and the go/no-go decision for
real money based on the pre-locked Phase 4 protocol.

---

## Verdict

**NO-GO for real money.** Never arm micro-live while Phase 4's pre-locked
protocol reports FAIL. The existing `MicroLiveGuard` enforces this by
construction (5.3 gate) — it refuses every entry until
`reports/phase4_paper.json` reports `verdict: "PASS"`.

## Why (evidence)

1. **Walk-forward shows no edge on the tunable universe.**
   - Phase 2 universe FAIL: deflated OOS Sharpe **-0.307** (needs > 0.05),
     PBO **0.4844** (needs < 0.05), after 126 trials. Detail:
     *"Deflated OOS Sharpe -0.307 <= 0 after 126 trials."*
   - Phase 4 paper (unpatched shipped organs, live-feed CSVs) FAIL on the
     same numbers: deflated edge **-0.3068**, PBO **0.4844**, pooled Sharpe
     -0.0456.
   - 17/23 tickers had fewer than 30 OOS trades → INSUFFICIENT DATA, which
     is a FAIL, not a pass.

2. **Safety rails held.** Phase 4 paper: 30 sessions observed (2026-07-27 →
   2026-09-04), 384 trades, **0 kill-switch days**, **0 daily-loss days**,
   **0 errors**. The latched kill switch is exercised and never fired — but
   that is a safety result, not an edge result.

3. **Ablation isolates the levers (Phase 3).** Baseline EV **-0.317R**
   (WR 19.2%, R:R 2.21, 397 trades) on the 23-ticker pool. Removal deltas:
   - `orderbook_imbalance` is the strongest ADDITIVE factor (+0.208R).
   - `momentum` +0.079R, `institutional_flow` +0.077R additive.
   - `vwap_deviation` (-0.074R) and `vpin` (-0.115R) are currently HARMFUL.
   - `stop_wide` (4×ATR) is the biggest single exit lever (WR 36% at
     EV -0.10R vs 19% at -0.32R); `stop_tight` (2×ATR) is worse than
     baseline.
   - `threshold_flat` (0.50) doubles trade count (807) with no EV gain —
     keep the 0.65 threshold.
   - `off_all_alpha` → 0 trades (sanity confirms alpha is the only signal
     path).

## Data honesty

- Committed fixtures are ~5 trading days per ticker, recorded on different
  windows (union = 30 dates). That is statistically thin; the protocol
  requires `MIN_TRADES=30` per ticker OOS and the universe fails it.
- `data/market_data` (254-ticker live feed) is real but not committed and
  is the source for deeper incubation once backfilled.

## What would flip this to GO

1. Phase 4 paper PASS on **deeper** data (protocol hash-pinned:
   window=50, folds=6, deflated OOS Sharpe > 0.05 AND PBO < 0.05,
   P1-P6 release criteria) — i.e. enough OOS window to escape
   INSUFFICIENT DATA.
2. Re-run Phase 2/3 on the deeper pool to confirm the factor ranking holds
   and re-validate the 4×ATR stop hypothesis before shipping an exit change.
3. Micro-live then arms automatically via the existing 5.3 gate; start at
   the smallest footprint (1 ticker, well under the $5k notional cap).

## Non-negotiable

- CI red is a signal: the R2/wf gates must stay strict. No weakening to get
  green.
- No monkeypatching of denylisted singletons; no new unproven organs live.
- Phase 5 guard performs no broker I/O; order routing stays with the
  existing `_protect`/`_guard` rails.