"""Scanner regression tests: big-cap inclusive universe, weighted dedup."""
from __future__ import annotations

from types import SimpleNamespace

from hanoon_prime.data.scanner import CODE_WEIGHTS, SCAN_CONFIGS, IBScanner


def _item(symbol: str, rank: int):
    contract = SimpleNamespace(symbol=symbol)
    return SimpleNamespace(
        contractDetails=SimpleNamespace(contract=contract), rank=rank
    )


def test_scan_configs_are_well_formed():
    for cfg in SCAN_CONFIGS.values():
        assert cfg["instrument"] == "STK"
        assert cfg["locationCode"] == "STK.US.MAJOR"
        assert cfg["scanCode"]
        assert cfg["abovePrice"] >= 3.0
        assert cfg["aboveVolume"] > 0


def test_dollar_volume_codes_present():
    codes = {cfg["scanCode"] for cfg in SCAN_CONFIGS.values()}
    assert "MOST_ACTIVE_USD" in codes
    assert "MOST_ACTIVE_AVG_USD" in codes


def test_code_weights_cover_all_configs():
    assert set(CODE_WEIGHTS) == set(SCAN_CONFIGS)


def test_dedup_prefers_dollar_volume_code():
    scanner = IBScanner(None)
    scanner._ingest_item("most_active", _item("TSLA", 1))
    scanner._ingest_item("most_active_usd", _item("TSLA", 5))
    assert scanner._results["TSLA"].rank == 5


def test_symbol_length_filter_drops_noise():
    scanner = IBScanner(None)
    scanner._ingest_item("most_active_usd", _item("ABC1234567", 1))
    assert "ABC1234567" not in scanner._results
