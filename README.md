# HANOON PRIME

A minimal, profit-first trading system. **IB → JULI → Entry/Hold/Exit + Learning.**

> No code merges without backtest proof. See [CONTRACT.md](CONTRACT.md).

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run backtest on historical data
python -m hanoon_prime.backtest --tickers AAPL,TSLA,SPY --data-dir /path/to/csvs

# Run safety net validation
pytest tests/test_contract.py -v
```

## Architecture

- **Alpha**: 5 indicators (VPIN, orderbook imbalance, institutional flow, momentum, VWAP deviation)
- **Scoring**: weighted average → normalized score [0.10, 0.70]
- **EV gate**: `p * R - (1-p)` must be > 0 after fee drag
- **Thinker**: `score ≥ 0.58 AND confidence > 0.50 AND direction ≠ 0 → ENTER`
- **Exit**: trailing stop + fixed target, measured by realized R:R
- **Learning**: ONE online weight gradient — off until proven edge

## Safety Nets (hard stops, enforced as exceptions)

- Max position size: $1,000
- Max loss per trade: $50
- Max concurrent positions: 3
- Daily loss limit: $200 → hard shutdown
- Emergency stop: 3 consecutive losses → pause 60 min

## Development Rules

All rules enforced in CI. See [CONTRACT.md](CONTRACT.md) for full contract.
