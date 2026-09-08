"""config.py — re-export shim for the brain-owned trading policy.

The decision configuration now lives in ``brain/policy/trading_policy.py``;
this module keeps the historical import path (telemetry, ib_adapter, tests)
working against the SAME singleton object.
"""

from __future__ import annotations

from .brain.policy.trading_policy import TRADING_CONFIG, TradingConfig

__all__ = ["TradingConfig", "TRADING_CONFIG"]
