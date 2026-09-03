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

from ..ib_compat import ib
from ..immune import IB_HOST, IB_PAPER_PORT

log = logging.getLogger(__name__)

# Scan configurations for different market conditions
SCAN_CONFIGS: dict[str, dict[str, Any]] = {
    "hot_volume": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "HOT_ACTIVE_US",
        "aboveVolume": 1_000_000,
    },
    "top_gainers": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "TOP_VOLUME_GAIN",
        "aboveVolume": 500_000,
    },
    "high_volatility": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "HOT_ACTIVE_US",
        "aboveVolume": 2_000_000,
    },
}


@dataclass
class ScanResult:
    """A single scanner result with contract info."""

    symbol: str
    contract: Any = None
    rank: int = 0
    distance: str = ""
    benchmark: str = ""
    projection: str = ""
    scan_name: str = ""
    discovered_at: float = field(default_factory=time.time)


class IBScanner:
    """IB Gateway market scanner — discovers trading candidates."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self._results: dict[str, ScanResult] = {}
        self._last_scan: float = 0.0
        self._scan_interval: float = 300.0  # 5 minutes
        self._active_subs: dict[int, str] = {}

    def scan(self, config_name: str = "hot_volume") -> list[ScanResult]:
        """Run a scanner subscription and collect results."""
        config = SCAN_CONFIGS.get(config_name, SCAN_CONFIGS["hot_volume"])
        try:
            from ib_insync import ScannerSubscription, TagValue

            sub = ScannerSubscription()
            sub.instrument = config["instrument"]
            sub.locationCode = config["locationCode"]
            sub.scanCode = config["scanCode"]
            sub.aboveVolume = config.get("aboveVolume", 1_000_000)
            sub.numberOfRows = 50
            filters = []
            if "aboveVolume" in config:
                filters.append(TagValue("volumeAbove", str(config["aboveVolume"])))
            req_id = int(time.time() * 1000) % 100000
            self.ib.reqScannerSubscription(req_id, sub, [], filters)
            self._active_subs[req_id] = config_name
            self._last_scan = time.time()
            log.info("Scanner started: %s (reqId=%d)", config_name, req_id)
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
            result = ScanResult(
                symbol=sym,
                contract=contract,
                rank=rank,
                distance=distance,
                benchmark=benchmark,
                projection=projection,
                scan_name=self._active_subs.get(req_id, ""),
            )
            self._results[sym] = result
        except Exception as e:
            log.debug("Scan data parse error: %s", e)

    def on_scan_end(self, req_id: int) -> None:
        """Handle scan completion callback."""
        scan_name = self._active_subs.pop(req_id, "unknown")
        count = len(self._results)
        log.info("Scanner '%s' complete: %d candidates", scan_name, count)

    def get_candidates(self) -> list[ScanResult]:
        """Return current scan results sorted by rank."""
        results = list(self._results.values())
        results.sort(key=lambda r: r.rank)
        return results

    def cancel_all(self) -> None:
        """Cancel all active scanner subscriptions."""
        for req_id in list(self._active_subs):
            try:
                self.ib.cancelScannerSubscription(req_id)
            except Exception as e:
                log.debug("cancel scan skip: %s", e)
        self._active_subs.clear()
        self._results.clear()

    def should_scan(self) -> bool:
        """Check if it's time for a new scan."""
        return time.time() - self._last_scan >= self._scan_interval
