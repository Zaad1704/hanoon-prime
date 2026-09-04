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
}


@dataclass
class ScanResult:
    """A single scanner result with contract info."""

    symbol: str
    contract: Any = None
    rank: int = 0
    discovered_at: float = field(default_factory=time.time)


class IBScanner:
    """IB Gateway market scanner — discovers trading candidates."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self._results: dict[str, ScanResult] = {}
        self._last_scan: float = 0.0
        self._scan_interval: float = 300.0
        self._scan_list: Any = None
        self._scan_started: float = 0.0

    def scan(self, config_name: str = "most_active") -> int:
        """Start a scanner subscription. Returns count (0=started async)."""
        config = SCAN_CONFIGS.get(config_name, SCAN_CONFIGS["most_active"])
        try:
            from ib_insync import ScannerSubscription

            self._cancel_scan()
            sub = ScannerSubscription(
                numberOfRows=50,
                instrument=config["instrument"],
                locationCode=config["locationCode"],
                scanCode=config["scanCode"],
            )
            self._scan_list = self.ib.reqScannerSubscription(sub)
            self._scan_started = time.time()
            self._last_scan = time.time()
            log.info("Scanner started: %s", config_name)
            return 0
        except Exception as e:
            log.warning("Scanner failed: %s", e)
            self._last_scan = time.time()
            return 0

    def collect(self) -> list[ScanResult]:
        """Poll the live ScanDataList for new results. Call each cycle."""
        if self._scan_list is None:
            return self.get_candidates()
        try:
            for item in self._scan_list:
                cd = getattr(item, "contractDetails", None)
                c = getattr(cd, "contract", None)
                if c is None:
                    continue
                sym = getattr(c, "symbol", "")
                if not sym or len(sym) > 6:
                    continue
                if sym not in self._results:
                    self._results[sym] = ScanResult(
                        symbol=sym, contract=c, rank=item.rank
                    )
        except Exception as e:
            log.debug("Scan collect error: %s", e)
        elapsed = time.time() - self._scan_started
        if elapsed > 10.0 and self._scan_list is not None:
            count = len(self._results)
            if count > 0:
                log.info("Scanner collected: %d candidates", count)
            self._scan_list = None
        return self.get_candidates()

    def get_candidates(self) -> list[ScanResult]:
        """Return current scan results sorted by rank."""
        results = list(self._results.values())
        results.sort(key=lambda r: r.rank)
        return results

    def _cancel_scan(self) -> None:
        """Cancel active scanner subscription without clearing results."""
        if self._scan_list is not None:
            try:
                self.ib.cancelScannerSubscription(self._scan_list)
            except Exception as e:
                log.debug("Scanner cancel error: %s", e)
            self._scan_list = None

    def cancel_all(self) -> None:
        """Cancel all active scanner subscriptions and clear results."""
        self._cancel_scan()
        self._results.clear()

    def should_scan(self) -> bool:
        """Check if it's time for a new scan."""
        return time.time() - self._last_scan >= self._scan_interval
