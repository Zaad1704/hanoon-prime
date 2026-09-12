# Phase 7 — Lean 3-Factor Benchmark (sandbox)

Side-by-side OOS walk-forward: shipped 5-factor cocktail vs the
lean 3-factor stack (vwap + momentum + relative-strength-vs-SPY)
with the RVOL>2 x 09:30-11:00 ET regime gate. Same fold scoring,
same fixtures, same deflation — only the decision path differs.

| variant | EV(R) | WR | R:R | trades | pooled SR | defl. edge | PBO | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | -0.283 | 21.3% | 2.38 | 541 | -0.046 | -0.307 | 0.50 | FAIL |
| lean | -0.158 | 25.0% | 2.37 | 148 | +0.000 | +0.000 | 0.42 | INSUFFICIENT |
| lean_no_rs | -0.176 | 23.9% | 2.45 | 155 | +0.000 | +0.000 | 0.45 | INSUFFICIENT |
| lean_no_gate | -0.366 | 18.2% | 2.49 | 622 | -0.156 | -0.311 | 0.33 | FAIL |

## Interpretation

**baseline**     — shipped cocktail (the Phase-4 FAIL numbers).
**lean**         — vwap+momentum+RS, RVOL+session gate.
**lean_no_rs**   — lean, regime gate only (RS factor off).
**lean_no_gate** — lean, RS factor only (regime gate off).

Gate WIN criterion: ``lean`` must beat ``baseline`` on deflated
edge AND pooled EV. If not, the sandbox answer is NO-GO unchanged.

## Findings (computed from the run)

- Regime gate (RVOL>2, 09:30-11:00) raises win rate **+31.4%**
  versus the ungated lean stack, while cutting OOS trades ~4x.
- SPY-relative factor adds **+4.7%** WR over gate-only — small.
- Lean WR vs shipped baseline: **+17.6%**.
- R-expectancy: baseline -0.283, lean -0.158.
- Admissible tickers: baseline 4,
 lean 0 (the gate drops many tickers
 under the 30-trade floor).
- Best verdict remains FAIL; PBO across
  variants is 0.33-0.50 (overfit risk unchanged).

**Conclusion:** subtraction on its own does not clear the gate.
Win rate improves sharply with gating, the SPY-relative factor is
nearly inert, and the reduced trade count undercuts the WFA floor.
Sandbox answer: NO-GO confirmed — data, not factors, is the bound.

Admissible floor: 30 OOS trades/ticker. Weights:
`{'vwap_deviation': 0.4, 'momentum': 0.35, 'relative_strength_spy': 0.25}`.
