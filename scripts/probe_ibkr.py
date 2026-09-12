#!/usr/bin/env python3
"""scripts/probe_ibkr.py — READ-ONLY feasibility probe for the IB Gateway API.

Answeres the Phase-10 vendor question ("can IBKR supply the multi-year 1-min
panel with pre-market coverage?") against the LIVE gateway the operator already
runs (default 127.0.0.1:4002 = IB Gateway paper). It only makes market-data
requests — never queries positions and never places orders.

Probes, per ticker:
  1. contract resolution + trading hours string (pre-market availability)
  2. reqHeadTimeStamp -> earliest data point reachable (raw depth signal)
  3. one 1-min reqHistoricalData window with useRTH=False -> bars returned,
     ET session breakdown (pre-market/RTH/after-hours) and volume presence
  4. pacing read: measured per-request latency -> extrapolated wall-clock for a
     full 60-month panel, cross-checked against IB's 60/10-min pacing rule

The point is empirical truth (like the Alpaca-news 404): subscription gaps,
delayed-only feeds and pacing limits surface here as concrete errors/numbers,
not doc claims.

Usage:
  python scripts/probe_ibkr.py
  python scripts/probe_ibkr.py --tickers AAPL,SPY --weeks 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock, util

_ET = ZoneInfo("America/New_York")
HOST, PORT, CLIENT_ID = "127.0.0.1", 4002, 7
_PRE, _RTH = (8 * 60, 9 * 60 + 29), (9 * 60 + 30, 16 * 60 - 1)
_REPO = Path(__file__).resolve().parent.parent


def _minute_of_day(dt: datetime) -> int:
    """Minutes since local midnight for session classification (ET wall time)."""
    return dt.hour * 60 + dt.minute


def _session(dt: datetime) -> str:
    """ET session tag: PRE / RTH / POST / OFF for a bar timestamp."""
    m = _minute_of_day(dt.astimezone(_ET))
    if _PRE[0] <= m <= _PRE[1]:
        return "PRE"
    if _RTH[0] <= m <= _RTH[1]:
        return "RTH"
    if 16 * 60 <= m <= 20 * 60 - 1:
        return "POST"
    return "OFF"


def _classify_bars(bars: list) -> dict:
    """{session: count} + first/last bar timestamps + volume-sample."""
    counts: dict[str, int] = {}
    style = util.df(bars)["date"]
    for bar in bars:
        counts[_session(bar.date)] = counts.get(_session(bar.date), 0) + 1
    sample = bars[-1] if bars else None
    return {
        "bars": len(bars),
        "sessions_et": counts,
        "first_et": bars[0].date.astimezone(_ET).isoformat() if bars else None,
        "last_et": bars[-1].date.astimezone(_ET).isoformat() if bars else None,
        "last_bar": None
        if sample is None
        else {
            "time_et": sample.date.astimezone(_ET).isoformat(),
            "close": float(sample.close),
            "volume": int(sample.volume),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IBKR read-only feasibility probe")
    parser.add_argument("--tickers", default="AAPL,MSFT,SPY,NVDA,VALE")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--client-id", type=int, default=CLIENT_ID)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ib = IB()
    t0 = time.monotonic()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=args.timeout)
    except Exception as exc:
        print(
            f"ERROR connect {args.host}:{args.port} (clientId={args.client_id}): {exc}"
        )
        print(
            "  -> ensure IB Gateway is running and 'Enable ActiveX and Socket Clients' is on"
        )
        return 1
    print(
        f"CONNECTED {args.host}:{args.port} clientId={args.client_id} "
        f"server={ib.client.serverVersion()}"
    )
    print(f"ACCOUNTS {list(ib.managedAccounts())} (DU... = paper, U... = live)")
    print(f"EQTIME  {ib.reqCurrentTime()}")

    errors: list[tuple[int, str]] = []
    ib.errorEvent += lambda reqId, code, msg, contract: errors.append((code, msg))

    duration = f"{args.weeks} W"
    results: dict[str, dict] = {}
    for ticker in tickers:
        contract = Stock(ticker, "SMART", "USD")
        ib.qualifyContracts(contract)
        cx = {}
        if contract:
            detail = ib.reqContractDetails(contract)
            cx["tradingHours"] = detail[0].tradingHours if detail else None
            header = ib.reqHeadTimeStamp(contract, "TRADES", False, 1)
            cx["headTimeStamp_et"] = (
                header.astimezone(_ET).isoformat() if header else None
            )
        else:
            cx["tradingHours"] = None
            cx["headTimeStamp_et"] = None
            cx["resolved"] = False
            results[ticker] = cx
            print(f"{ticker:6s} NOT-RESOLVED")
            continue
        cx["resolved"] = True
        hist_start = time.monotonic()
        bars = ib.reqHistoricalData(
            contract, "", duration, "1 min", "TRADES", False, 1, False, []
        )
        latency_ms = int((time.monotonic() - hist_start) * 1000)
        cx.update(_classify_bars(bars))
        cx["latency_ms"] = latency_ms
        results[ticker] = cx
        print(
            f"{ticker:6s} bars={len(bars):5d} "
            f"sessions={dict(sorted(cx.get('sessions_et', {}).items()))} "
            f"latency={latency_ms}ms headTS={cx['headTimeStamp_et']}"
        )
        time.sleep(2.0)

    months_full = 60
    paced_per_window = 60
    # IB bounds historical requests to 60 in any 10-minute window.
    requests = months_full * len(tickers)
    windows = max(1, (requests + paced_per_window - 1) // paced_per_window)
    est_min = windows * 10
    summary = {
        "probed_at_utc": datetime.now().astimezone().isoformat(),
        "host": args.host,
        "port": args.port,
        "client_id": args.client_id,
        "tickers": tickers,
        "errors": errors,
        "full_panel_cost_estimate": {
            "months_1min_per_ticker_each_1_req": months_full,
            "requests_total": requests,
            "ib_pacing": "60 req / 10 min",
            "est_wallclock_min": est_min,
            "assumes_no_errors_and_continuous_paging": True,
        },
        "latest_probe_duration_weeks": args.weeks,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    probe_dir = _REPO / "data" / "research"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "ibkr_probe.json").write_text(
        json.dumps({"summary": summary, "tickers": results}, indent=2) + "\n"
    )
    # console side by side
    print("\n--- ticker probe table ---")
    for ticker, cx in results.items():
        print(
            f"{ticker:6s} resolved={cx.get('resolved')} "
            f"bars={cx.get('bars', '-')} "
            f"headTS={str(cx.get('headTimeStamp_et'))[:10]}"
        )
    if errors:
        print("\n--- IB errors observed ---")
        for code, msg in errors:
            print(f"  {code}: {msg}")
    print(
        f"\nFull-panel estimate: {requests} requests ~ {est_min} min "
        f"(60 req/10 min pacing budget)"
    )
    print(f"probe JSON: data/research/ibkr_probe.json")
    print(f"total elapsed: {summary['elapsed_sec']}s")
    ib.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
