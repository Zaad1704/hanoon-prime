"""hanoon_prime.data.scanner — IB Gateway market scanner.

Discovers trading candidates using IB's built-in scanner.
Replaces hardcoded ticker lists with dynamic discovery.

Scanner returns ScanDataList that auto-populates via events.
We poll the list to extract results. Max 50 results per scan.
Raw-by-doctrine: results stay unranked/unfiltered — the brain decides.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..ib_compat import _ib_available
from ..ib_compat import ib as _ib

log = logging.getLogger(__name__)
DATA_INSTRUMENT: str = "STK"
DATA_LOCATION: str = "STK.US.MAJOR"
# 10 all-cap codes fill IB's 10-scan limit; each returns 50 rows.
SCAN_CONFIGS: dict[str, str] = {
    "most_active_usd": "MOST_ACTIVE_USD",
    "most_active_avg_usd": "MOST_ACTIVE_AVG_USD",
    "top_price_range": "TOP_PRICE_RANGE",
    "most_active": "MOST_ACTIVE",
    "top_gainers": "TOP_PERC_GAIN",
    "top_losers": "TOP_PERC_LOSE",
    "market_cap_desc": "MARKET_CAP_USD_DESC",
    "market_cap_asc": "MARKET_CAP_USD_ASC",
    "hot_by_volume": "HOT_BY_VOLUME",
    "top_volume_rate": "TOP_VOLUME_RATE",
}
# Frozen ScanCode whitelist (R25/R26): every scanCode belongs here.
_ALLOWED_CODES: tuple[str, ...] = (
    "MOST_ACTIVE_USD",
    "MOST_ACTIVE_AVG_USD",
    "TOP_PRICE_RANGE",
    "MOST_ACTIVE",
    "TOP_PERC_GAIN",
    "TOP_PERC_LOSE",
    "MARKET_CAP_USD_ASC",
    "MARKET_CAP_USD_DESC",
    "HOT_BY_VOLUME",
    "TOP_VOLUME_RATE",
)
ALLOWED_SCANCODES: frozenset[str] = frozenset(_ALLOWED_CODES)


@dataclass
class ScanResult:
    """A single scanner result with contract info."""

    symbol: str
    contract: Any = None
    rank: int = 0
    discovered_at: float = field(default_factory=time.time)


class IBScanner:
    """IB Gateway market scanner — discovers trading candidates.

    Subscribes to multiple scan codes simultaneously and deduplicates
    by keeping the best rank per ticker across all scans.
    """

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self._results: dict[str, ScanResult] = {}
        self._last_scan: float = 0.0
        self._scan_interval: float = 300.0
        self._scan_lists: dict[str, Any] = {}
        self._scan_started: float = 0.0

    def scan(self, config_name: str = "all") -> int:
        """Start scanner subscriptions. 'all' subscribes to all 10 codes."""
        if not _ib_available:
            return 0
        self._cancel_scan()
        configs = (
            SCAN_CONFIGS
            if config_name == "all"
            else {config_name: SCAN_CONFIGS.get(config_name, "MOST_ACTIVE")}
        )
        for name, scan_code in configs.items():
            self._start_sub(name, scan_code)
        self._scan_started = time.time()
        self._last_scan = time.time()
        log.info("Scanner started: %d codes", len(self._scan_lists))
        return 0

    def _start_sub(self, name: str, scan_code: str) -> None:
        """Start one scanner subscription — no price/volume filters."""
        try:
            sub = _ib.ScannerSubscription(
                numberOfRows=50,
                instrument=DATA_INSTRUMENT,
                locationCode=DATA_LOCATION,
                scanCode=scan_code,
            )
            self._scan_lists[name] = self.ib.reqScannerSubscription(sub)
        except Exception as e:
            log.debug("Scanner subscribe failed for %s: %s", name, e)

    def collect(self) -> list[ScanResult]:
        """Poll all scanner subscriptions, dedup by best rank."""
        if not self._scan_lists:
            return self.get_candidates()
        self._drain_scan_lists()
        self._maybe_finalize()
        return self.get_candidates()

    def _drain_scan_lists(self) -> None:
        """Drain scan results into deduped dict."""
        try:
            for scan_list in list(self._scan_lists.values()):
                for item in scan_list:
                    self._ingest_item(item)
        except Exception as e:
            log.debug("Scan collect error: %s", e)

    def _ingest_item(self, item: Any) -> None:
        """Ingest one scan item at its natural rank — no offsets."""
        cd = getattr(item, "contractDetails", None)
        c = getattr(cd, "contract", None)
        if c is None:
            return
        sym = getattr(c, "symbol", "")
        if not sym or len(sym) > 6:
            return
        eff = item.rank
        existing = self._results.get(sym)
        if existing is None or eff < existing.rank:
            self._results[sym] = ScanResult(symbol=sym, contract=c, rank=eff)

    def _maybe_finalize(self) -> None:
        """Clear scan lists after timeout."""
        elapsed = time.time() - self._scan_started
        if elapsed > 10.0 and self._scan_lists:
            if self._results:
                log.info("Scanner collected: %d candidates", len(self._results))
            self._scan_lists.clear()

    def get_candidates(self) -> list[ScanResult]:
        """Return current scan results sorted by rank."""
        results = list(self._results.values())
        results.sort(key=lambda r: r.rank)
        return results

    def _cancel_scan(self) -> None:
        """Cancel all active scanner subscriptions."""
        for name, scan_list in list(self._scan_lists.items()):
            try:
                self.ib.cancelScannerSubscription(scan_list)
            except Exception as e:
                log.debug("Scanner cancel error: %s", e)
        self._scan_lists.clear()

    def cancel_all(self) -> None:
        """Cancel all active scanner subscriptions and clear results."""
        self._cancel_scan()
        self._results.clear()

    def should_scan(self) -> bool:
        """Check if it's time for a new scan."""
        return time.time() - self._last_scan >= self._scan_interval


__all__ = ["SCAN_CONFIGS", "ALLOWED_SCANCODES", "IBScanner", "ScanResult"]
