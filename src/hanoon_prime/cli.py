"""hanoon_prime.cli — Live trading entry point.

Starts the IB Streaming Bot (IB Gateway port 4002 paper) on the
FAST tickers, plus a lightweight TelemetryAPI on :8080 for the
cloudflared tunnel (api.hanoonweb.xyz → :8080).

Usage:
    python3 -m hanoon_prime.cli           # default FAST tickers
    python3 -m hanoon_prime.cli AAPL NVDA # custom tickers
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from .ib_adapter import IBStreamingBot
from .immune import FAST_TICKERS
from .telemetry import TelemetryAPI

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging — suppress verbose ib_insync messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress verbose ib_insync.wrapper messages (portfolio updates, etc.)
    logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
    # Suppress ib_insync internal messages
    logging.getLogger("ib_insync.ib").setLevel(logging.WARNING)


def main() -> None:
    """Entry point: start telemetry + bot, connect to IB Gateway live."""
    _setup_logging()
    tickers = sys.argv[1:] if len(sys.argv) > 1 else list(FAST_TICKERS)
    journal_path = Path("runtime") / "journal_live.jsonl"
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
