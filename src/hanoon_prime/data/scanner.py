"""hanoon_prime.data.scanner — IB Gateway market scanner.

Discovers trading candidates using IB's built-in scanner.
Replaces hardcoded ticker lists with dynamic discovery.

Scanner returns contracts only — no market data.
Must call reqMktData separately for bid/ask/last.
Max 50 results per scan, 10 active scans.
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
    scan_name: str = ""
    discovered_at: float = field(default_factory=time.time)


class IBScanner:
    """IB Gateway market scanner — discovers trading candidates."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self._results: dict[str, ScanResult] = {}
        self._last_scan: float = 0.0
        self._scan_interval: float = 300.0
        self._scan_list: Any = None

    def scan(self, config_name: str = "hot_volume") -> list[ScanResult]:
        """Run a scanner subscription and collect results."""
        config = SCAN_CONFIGS.get(config_name, SCAN_CONFIGS["hot_volume"])
        try:
            from ib_insync import ScannerSubscription, TagValue

            sub = ScannerSubscription()
            sub.instrument = config["instrument"]
            sub.locationCode = config["locationCode"]
            sub.scanCode = config["scanCode"]
            sub.numberOfRows = 50
            self._scan_list = self.ib.reqScannerSubscription(sub, [], [])
            self._last_scan = time.time()
            log.info("Scanner started: %s", config_name)
            return []
        except Exception as e:
            log.warning("Scanner failed: %s", e)
            return []

    def on_scan_data(
        self,
        req_id: int,
        rank: int,
        contract_details: Any,
        distance: str,
        benchmark: str,
        projection: str,
        legs: str,
    ) -> None:
        """Handle scanner data callback from IB."""
        try:
            contract = contract_details.contract
            sym = contract.symbol
            self._results[sym] = ScanResult(
                symbol=sym,
                contract=contract,
                rank=rank,
            )
        except Exception as e:
            log.debug("Scan data parse error: %s", e)

    def on_scan_end(self, req_id: int) -> None:
        """Handle scan completion callback."""
        count = len(self._results)
        log.info("Scanner complete: %d candidates", count)

    def get_candidates(self) -> list[ScanResult]:
        """Return current scan results sorted by rank."""
        results = list(self._results.values())
        results.sort(key=lambda r: r.rank)
        return results

    def cancel_all(self) -> None:
        """Cancel all active scanner subscriptions."""
        if self._scan_list:
            try:
                self.ib.cancelScannerSubscription(self._scan_list)
            except Exception as e:
                log.debug("cancel scan skip: %s", e)
            self._scan_list = None
        self._results.clear()

    def should_scan(self) -> bool:
        """Check if it's time for a new scan."""
        return time.time() - self._last_scan >= self._scan_interval
