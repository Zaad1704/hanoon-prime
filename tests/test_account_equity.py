"""Equity fallback chain (pre-market sizing)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanoon_prime.account_equity import (  # noqa: E402
    load_equity_cache,
    resolve_account_equity,
    save_equity_cache,
)


def _ib(net: str | None = None, cash: str = "0.0", positions: tuple = ()) -> MagicMock:
    ib = MagicMock()
    items = []
    if net is not None:
        items.append(SimpleNamespace(tag="NetLiquidation", value=net))
    items.append(SimpleNamespace(tag="CashBalance", value=cash))
    ib.accountSummary.return_value = items

    def _porfolio() -> tuple:
        return positions

    ib.portfolio.return_value = positions  # type: ignore[attr-defined]
    return ib


def _pos(sym: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=sym),
        marketValue=value,
        position=10,
        unrealizedPNL=1.0,
    )


class TestEquityFallback:
    def test_net_liq_is_used_and_cached(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(tmp_path / "eq.json"))
        ib = _ib(net="100000")
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq == 100000.0
        assert synced is True
        assert load_equity_cache() == 100000.0

    def test_local_cash_plus_portfolio(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(tmp_path / "eq.json"))
        ib = _ib(cash="500", positions=(_pos("AAPL", 100.0),))
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq == 600.0
        assert synced is True

    def test_cache_fallback(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / "eq.json"
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(p))
        save_equity_cache(1234.5, p)
        ib = _ib()  # empty summary, empty portfolio
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq == 1234.5
        assert synced is True

    def test_unknown_stays_unsynced(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(tmp_path / "eq.json"))
        ib = _ib()
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq is None
        assert synced is False

    def test_bad_cache_file_ignored(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / "eq.json"
        p.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(p))
        ib = _ib()
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq is None
        assert synced is False
