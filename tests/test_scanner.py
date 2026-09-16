"""Scanner regression tests: raw all-cap discovery, natural-rank dedup."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hanoon_prime.data.budget import DataBudget
from hanoon_prime.data.scanner import ALLOWED_SCANCODES, SCAN_CONFIGS, IBScanner

SRC = Path(__file__).resolve().parent.parent / "src"


def _item(symbol: str, rank: int):
    contract = SimpleNamespace(symbol=symbol)
    return SimpleNamespace(
        contractDetails=SimpleNamespace(contract=contract), rank=rank
    )


def test_scan_configs_are_raw_and_well_formed():
    codes = set(SCAN_CONFIGS.values())
    assert codes <= ALLOWED_SCANCODES, "R26: configured codes must be whitelisted"
    assert len(codes) == 10, "IB's 10-scan subscription limit must be used"
    assert len(SCAN_CONFIGS) <= 10
    for code in codes:
        assert code.isupper(), "codes must be ALL-CAPS"


def test_real_ib_codes_used_not_typoed():
    codes = set(SCAN_CONFIGS.values())
    assert "TOP_PERC_GAIN" in codes, "real IB code is TOP_PERC_GAIN, not TOP_PCT_GAIN"
    assert "TOP_PERC_LOSE" in codes, "real IB code is TOP_PERC_LOSE, not TOP_PCT_LOSE"
    assert "MOST_ACTIVE_USD" in codes
    assert "MOST_ACTIVE_AVG_USD" in codes


def test_all_cap_coverage_raw_discovery_codes():
    codes = set(SCAN_CONFIGS.values())
    assert "MARKET_CAP_USD_ASC" in codes, "small-cap names must be discoverable"
    assert "HOT_BY_VOLUME" in codes
    assert "TOP_VOLUME_RATE" in codes


def test_dedup_keeps_natural_rank_no_weights():
    scanner = IBScanner(None)
    scanner._ingest_item(_item("TSLA", 1))
    scanner._ingest_item(_item("TSLA", 5))
    assert scanner._results["TSLA"].rank == 1, "best (lowest) natural rank wins"


def test_no_scanner_side_filters():
    src = (SRC / "hanoon_prime" / "data" / "scanner.py").read_text()
    for token in ("CODE_WEIGHTS", "abovePrice", "aboveVolume"):
        assert token not in src, f"raw doctrine forbids {token}"


def test_symbol_length_filter_drops_noise():
    scanner = IBScanner(None)
    scanner._ingest_item(_item("ABC1234567", 1))
    assert "ABC1234567" not in scanner._results


def test_get_candidates_sorted_by_natural_rank():
    scanner = IBScanner(None)
    scanner._ingest_item(_item("AAA", 5))
    scanner._ingest_item(_item("BBB", 2))
    assert [c.symbol for c in scanner.get_candidates()] == ["BBB", "AAA"]


def test_budget_rotates_small_pool_fully():
    budget = DataBudget()
    to_sub, _ = budget.allocate(set(), ["A", "B", "C", "D"])
    assert set(to_sub) == {"A", "B", "C", "D"}


def test_budget_rotation_admits_fresh_names_over_time():
    budget = DataBudget()
    budget.allocate(set(), ["A"])
    fresh = [f"T{i}" for i in range(300)]
    for _ in range(10):
        budget.allocate(set(), ["A"] + fresh)
    tracked = budget.get_all_tracked()
    assert any("T" in t for t in tracked), "rotation must stream fresh discovery names"
    assert budget.count_tiers().get("L1", 0) <= 100, "IB L1 line allowance respected"


def test_positions_always_keep_tbt_seat():
    budget = DataBudget()
    budget.allocate({"POS"}, [])
    assert budget.get_tbt_tickers() == ["POS"]
    budget.allocate({"POS"}, [f"C{i}" for i in range(90)])
    assert "POS" in budget.get_tbt_tickers(), "position seat never rotated out"
