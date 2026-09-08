"""hanoon_prime.cli — Live trading entry point.

Starts the IB Streaming Bot (IB Gateway port 4002 paper) on the
FAST tickers, plus a lightweight TelemetryAPI on :8080 for the
cloudflared tunnel (api.hanoonweb.xyz → :8080).

Usage:
    python3 -m hanoon_prime.cli           # auto-seed from liquid universe
    python3 -m hanoon_prime.cli AAPL NVDA # explicit tickers + scanner discovery
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from .ib_adapter import IBStreamingBot
from .immune import LIQUID_US_SEED
from .telemetry import TelemetryAPI

log = logging.getLogger(__name__)


class _CleanFormatter(logging.Formatter):
    """Strip the package prefix from logger names and format cleanly.

    hanoon_prime.juli → juli, hanoon_prime.ib_cycle → ib_cycle, etc.
    """

    _PKG = "hanoon_prime."

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        if name.startswith(self._PKG):
            name = name[len(self._PKG):]
        record.name = name
        return super().format(record)


def _setup_logging() -> None:
    """Configure logging with clean formatting and noise suppression."""
    fmt = _CleanFormatter(
        fmt="%(asctime)s.%(msecs)03d  %(levelname)-5s  %(name)-16s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    # ib_insync noise: suppress verbose wrappers + cancelMktData Ticker dumps
    logging.getLogger("ib_insync.wrapper").setLevel(logging.CRITICAL)
    logging.getLogger("ib_insync.ib").setLevel(logging.CRITICAL)
    logging.getLogger("ib_insync.client").setLevel(logging.WARNING)


def main() -> None:
    """Entry point: start telemetry + bot, connect to IB Gateway live."""
    _setup_logging()
    # CLI args: explicit tickers, or auto-seed from liquid universe
    tickers = (
        sys.argv[1:] if len(sys.argv) > 1 else list(LIQUID_US_SEED)[:5]
    )  # top 5 for speed
    # Anchor to repo root, NOT CWD — bot may be launched from anywhere.
    repo_root = Path(__file__).resolve().parents[2]
    journal_path = repo_root / "runtime" / "journal_live.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    bot = IBStreamingBot(account="PAPER")
    # Start health endpoint BEFORE connecting — Cloudflare tunnel can
    # reach /health while the bot is still establishing IB session.
    telemetry = TelemetryAPI(bot, journal_path)
    telemetry.start()

    try:
        bot.run_paper(tickers)
    except Exception:
        log.error("Bot crashed:\n%s", traceback.format_exc())
        raise
    finally:
        telemetry.stop()


if __name__ == "__main__":
    main()
