# Phase 7 — Lean 3-Factor Benchmark (sandbox)

Side-by-side OOS walk-forward: shipped 5-factor cocktail vs the
lean 3-factor stack (vwap + momentum + relative-strength-vs-SPY)
with the RVOL>2 x 09:30-11:00 ET regime gate. Same fold scoring,
same fixtures, same deflation — only the decision path differs.

| variant | EV(R) | WR | R:R | trades | pooled SR | defl. edge | PBO | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | -0.239 | 21.4% | 2.55 | 10015 | -0.126 | -0.158 | 0.25 | FAIL |
| lean | -0.116 | 23.7% | 2.73 | 3937 | -0.058 | -0.109 | 0.66 | FAIL |
| lean_no_rs | -0.110 | 23.9% | 2.73 | 4009 | -0.055 | -0.104 | 0.42 | FAIL |
| lean_no_gate | -0.274 | 20.6% | 2.52 | 17919 | -0.151 | -0.175 | 0.53 | FAIL |

## Interpretation

**baseline**     — shipped cocktail (the Phase-4 FAIL numbers).
**lean**         — vwap+momentum+RS, RVOL+session gate.
**lean_no_rs**   — lean, regime gate only (RS factor off).
**lean_no_gate** — lean, RS factor only (regime gate off).

Gate WIN criterion: ``lean`` must beat ``baseline`` on deflated
edge AND pooled EV. If not, the sandbox answer is NO-GO unchanged.

## Findings (computed from the run)

- Regime gate (RVOL>2, 09:30-11:00) raises win rate **+15.8%**
  versus the ungated lean stack, while cutting OOS trades ~4x.
- SPY-relative factor adds **-0.8%** WR over gate-only — small.
- Lean WR vs shipped baseline: **+10.4%**.
- Pooled OOS R-expectancy (ALL tickers): baseline -0.239
 (10015 trades), lean -0.116
 (3937 trades).
- Pooled OOS R-expectancy (admissible only): baseline
 -0.239, lean -0.116
 (3937 trades, 22
 tickers) — the protocol pools only tickers with 30+ OOS
 trades, so all-ticker and admissible EV can diverge.
- Protocol verdict: lean FAIL
 (deflated edge -0.109, PBO
 0.66); baseline FAIL
 (deflated edge -0.158).

**Conclusion:** subtraction does not clear the gate on this set; sandbox answer remains NO-GO (data, not factors, is the bound).

Admissible floor: 30 OOS trades/ticker. Weights:
`{'vwap_deviation': 0.4, 'momentum': 0.35, 'relative_strength_spy': 0.25}`.
