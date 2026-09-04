"""hanoon_prime.data.scanner — IB Gateway market scanner.

Discovers trading candidates using IB's built-in scanner.
Replaces hardcoded ticker lists with dynamic discovery.

Scanner returns ScanDataList that auto-populates via events.
We poll the list to extract results. Max 50 results per scan.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)
SCAN_CONFIGS: dict[str, dict[str, Any]] = {
    "most_active": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "MOST_ACTIVE",
    },
    "top_gainers": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "TOP_PCT_GAIN",
    },
    "high_volume": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "HOT_BY_VOLUME",
    },
    "top_losers": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "TOP_PCT_LOSE",
    },
}


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
        """Start scanner subscriptions. 'all' subscribes to all 4 codes."""
        try:
            from ib_insync import ScannerSubscription

            self._cancel_scan()
            configs = (
                SCAN_CONFIGS
                if config_name == "all"
                else {
                    config_name: SCAN_CONFIGS.get(
                        config_name, SCAN_CONFIGS["most_active"]
                    )
                }
            )
            for name, config in configs.items():
                try:
                    sub = ScannerSubscription(
                        numberOfRows=50,
                        instrument=config["instrument"],
                        locationCode=config["locationCode"],
                        scanCode=config["scanCode"],
                    )
                    self._scan_lists[name] = self.ib.reqScannerSubscription(sub)
                except Exception as e:
                    log.debug("Scanner subscribe failed for %s: %s", name, e)
            self._scan_started = time.time()
            self._last_scan = time.time()
            log.info("Scanner started: %d codes", len(self._scan_lists))
            return 0
        except Exception as e:
            log.warning("Scanner failed: %s", e)
            self._last_scan = time.time()
            return 0

    def collect(self) -> list[ScanResult]:
        """Poll all scanner subscriptions, dedup by best rank."""
        if not self._scan_lists:
            return self.get_candidates()
        try:
            for name, scan_list in list(self._scan_lists.items()):
                for item in scan_list:
                    cd = getattr(item, "contractDetails", None)
                    c = getattr(cd, "contract", None)
                    if c is None:
                        continue
                    sym = getattr(c, "symbol", "")
                    if not sym or len(sym) > 6:
                        continue
                    existing = self._results.get(sym)
                    if existing is None or item.rank < existing.rank:
                        self._results[sym] = ScanResult(
                            symbol=sym, contract=c, rank=item.rank
                        )
        except Exception as e:
            log.debug("Scan collect error: %s", e)
        elapsed = time.time() - self._scan_started
        if elapsed > 10.0 and self._scan_lists:
            count = len(self._results)
            if count > 0:
                log.info("Scanner collected: %d candidates", count)
            self._scan_lists.clear()
        return self.get_candidates()

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
