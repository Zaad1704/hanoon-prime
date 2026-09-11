# Phase 2 — Walk-Forward Validation Report

Universe verdict: **FAIL**
- Pooled OOS Sharpe: `-0.0456`
- Deflated edge: `-0.3068`
- PBO (prob. backtest overfitting): `0.4844`
- Min OOS trades floor: `30`

Detail: Deflated OOS Sharpe -0.307 <= 0 after 126 trials.

## Per-ticker OOS (admissible only)

| ticker | oos trades | EV/trade | sharpe | verdict |
|---|---|---|---|---|
| INTC | 30 | -0.0015 | -0.2123 | FAIL |
| MARA | 38 | 0.0017 | 0.1528 | PASS |
| RDDT | 41 | -0.0007 | -0.1027 | FAIL |
| VALE | 30 | -0.0014 | -0.4057 | FAIL |

## Per-ticker raw OOS trade counts
- AAPL: 25 OOS trades
- CRWD: 22 OOS trades
- DDOG: 23 OOS trades
- F: 26 OOS trades
- INTC: 30 OOS trades
- IREN: 21 OOS trades
- MARA: 38 OOS trades
- MRNA: 23 OOS trades
- MSFT: 25 OOS trades
- NET: 29 OOS trades
- NIO: 13 OOS trades
- NVDA: 8 OOS trades
- PFE: 24 OOS trades
- PLTR: 24 OOS trades
- RDDT: 41 OOS trades
- SNAP: 24 OOS trades
- SNOW: 23 OOS trades
- SOFI: 18 OOS trades
- SOUN: 28 OOS trades
- SPY: 28 OOS trades
- T: 24 OOS trades
- TSLA: 22 OOS trades
- VALE: 30 OOS trades
