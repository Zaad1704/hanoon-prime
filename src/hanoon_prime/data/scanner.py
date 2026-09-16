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

from ..ib_compat import _ib_available
from ..ib_compat import ib as _ib

log = logging.getLogger(__name__)
# IB MOST_ACTIVE ranks by SHARES (floods with penny names); the dollar-volume
# codes surface big caps. Filters drop sub-$3 / <250k-share noise.
SCAN_CONFIGS: dict[str, dict[str, Any]] = {
    "most_active_usd": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "MOST_ACTIVE_USD",
        "abovePrice": 3.0,
        "aboveVolume": 250_000,
    },
    "most_active_avg_usd": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "MOST_ACTIVE_AVG_USD",
        "abovePrice": 3.0,
        "aboveVolume": 250_000,
    },
    "top_price_range": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "TOP_PRICE_RANGE",
        "abovePrice": 3.0,
        "aboveVolume": 250_000,
    },
    "most_active": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "MOST_ACTIVE",
        "abovePrice": 3.0,
        "aboveVolume": 250_000,
    },
    "top_gainers": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "TOP_PCT_GAIN",
        "abovePrice": 3.0,
        "aboveVolume": 250_000,
    },
    "top_losers": {
        "instrument": "STK",
        "locationCode": "STK.US.MAJOR",
        "scanCode": "TOP_PCT_LOSE",
        "abovePrice": 3.0,
        "aboveVolume": 250_000,
    },
}
# Rank offsets prefer dollar-volume names over share-volume penny leaders.
CODE_WEIGHTS: dict[str, int] = {
    "most_active_usd": 0,
    "most_active_avg_usd": 0,
    "top_price_range": 50,
    "top_gainers": 60,
    "top_losers": 70,
    "most_active": 90,
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
        if not _ib_available:
            return 0
        self._cancel_scan()
        configs = (
            SCAN_CONFIGS
            if config_name == "all"
            else {
                config_name: SCAN_CONFIGS.get(config_name, SCAN_CONFIGS["most_active"])
            }
        )
        for name, config in configs.items():
            self._start_sub(name, config)
        self._scan_started = time.time()
        self._last_scan = time.time()
        log.info("Scanner started: %d codes", len(self._scan_lists))
        return 0

    def _start_sub(self, name: str, config: dict[str, Any]) -> None:
        """Start a single scanner subscription."""
        try:
            kwargs: dict[str, Any] = {
                "numberOfRows": 50,
                "instrument": config["instrument"],
                "locationCode": config["locationCode"],
                "scanCode": config["scanCode"],
            }
            for key in ("abovePrice", "aboveVolume"):
                if config.get(key):
                    kwargs[key] = config[key]
            sub = _ib.ScannerSubscription(**kwargs)
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
            for name, scan_list in list(self._scan_lists.items()):
                for item in scan_list:
                    self._ingest_item(name, item)
        except Exception as e:
            log.debug("Scan collect error: %s", e)

    def _ingest_item(self, name: str, item: Any) -> None:
        """Ingest a single scan item (weighted across scan codes)."""
        cd = getattr(item, "contractDetails", None)
        c = getattr(cd, "contract", None)
        if c is None:
            return
        sym = getattr(c, "symbol", "")
        if not sym or len(sym) > 6:
            return
        eff = item.rank + CODE_WEIGHTS.get(name, 0)
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
