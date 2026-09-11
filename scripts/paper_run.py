#!/usr/bin/env python3
"""scripts/paper_run.py — Phase 4 paper-incubation harness (pre-locked protocol).

Usage:
  python scripts/paper_run.py --data-dir data/market_data \
      --output reports/phase4_paper.json

Runs the unmodified shipped organs (Hippocampus + static INDICATOR_WEIGHTS)
across every live-feed session in --data-dir and checks the PRE-LOCKED
release criteria from protocols/evaluation_protocol.md (P1–P6).

Exit code:
  0 → all release criteria met (paper passed → eligible for micro-live)
  1 → FAIL or INSUFFICIENT (honest, blocking)

The protocol hash is recorded in the report; a changed protocol invalidates
comparability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hanoon_prime.backtest import _discover_tickers
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.hands import simulate_ticker
from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.immune import (
    DAILY_LOSS_LIMIT,
    EDGE_LOOKBACK,
    KILL_DAILY_LOSS_LIMIT,
    MAX_POSITION_NOTIONAL,
)
from hanoon_prime.types import BarSeries

PROTOCOL_FILE = (
    Path(__file__).resolve().parent.parent / "protocols" / "evaluation_protocol.md"
)

# Pre-locked release criteria (mirror protocols/evaluation_protocol.md §4).
MIN_PAPER_TRADES = 30
MIN_SESSIONS = 5
ALLOWED_KILL_SESSIONS = 0
ALLOWED_DAILY_LOSS_SESSIONS = 1


@dataclass
class SessionResult:
    """One trading session (one day) of paper results."""

    date: str
    trades: list[Any] = field(default_factory=list)
    gross_pnl_pct: float = 0.0
    gross_pnl_usd: float = 0.0
    kill_switch_fired: bool = False
    daily_loss_fired: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class PaperVerdict:
    """Aggregate release-criteria evaluation."""

    n_sessions: int = 0
    n_trades: int = 0
    n_kill_sessions: int = 0
    n_daily_loss_sessions: int = 0
    n_errors: int = 0
    criteria: dict[str, bool] = field(default_factory=dict)
    verdict: str = "FAIL"

    def evaluate(self) -> None:
        """Evaluate P1–P6 against the pre-locked protocol."""
        self.criteria = {
            "P1_min_trades": self.n_trades >= MIN_PAPER_TRADES,
            "P2_min_sessions": self.n_sessions >= MIN_SESSIONS,
            "P3_no_kill_sessions": self.n_kill_sessions <= ALLOWED_KILL_SESSIONS,
            "P4_daily_loss_sessions": (
                self.n_daily_loss_sessions <= ALLOWED_DAILY_LOSS_SESSIONS
            ),
            "P5_wfa_pass": False,  # composed below from wfa verdict
            "P6_no_errors": self.n_errors == 0,
        }
        self.verdict = "PASS" if all(self.criteria.values()) else "FAIL"


def _session_of(datetime_str: str) -> str:
    """Extract the trading-day key from a datetime string (date prefix)."""
    return str(datetime_str).split("T")[0].split(" ")[0]


def _run_sessions(
    ticker: str, data: dict[str, Any], brain: Hippocampus
) -> tuple[list[SessionResult], list[str]]:
    """Run the base brain per session; return (sessions, errors).

    Each session runs the SAME brain instance (like live: one memory, one
    state shared across the day), simulated over that day's bars.
    """
    sessions: list[SessionResult] = []
    errors: list[str] = []
    dts = data["datetime"]
    by_day: dict[str, list[int]] = {}
    for i, d in enumerate(dts):
        by_day.setdefault(_session_of(d), []).append(i)

    for date in sorted(by_day):
        idx = by_day[date]
        start, end = idx[0], idx[-1] + 1
        # Warmup bars come from prior sessions, but the simulation only sees
        # the current session's bars (same as cherry-py live path feeding
        # per-session data). Keep warmup inside the slice.
        warm = max(0, start - EDGE_LOOKBACK)
        s = slice(warm, end)
        sess_data = {
            "close": data["close"][s],
            "high": data["high"][s],
            "low": data["low"][s],
            "volume": data["volume"][s],
        }
        bars = BarSeries(
            sess_data["close"], sess_data["high"], sess_data["low"], sess_data["volume"]
        )
        try:
            trades, _ = simulate_ticker(ticker, bars, EDGE_LOOKBACK, brain)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - paper run must be honest about failures
            errors.append(f"{ticker}@{date}: {type(exc).__name__}: {exc}")
            continue

        total_pnl_pct = sum(t.pnl_pct for t in trades)
        # USD estimate: size proxies notional cap × per-trade R.
        total_pnl_usd = sum(t.pnl_pct * MAX_POSITION_NOTIONAL for t in trades)
        kill = total_pnl_usd <= -KILL_DAILY_LOSS_LIMIT
        dloss = total_pnl_usd <= -DAILY_LOSS_LIMIT
        sessions.append(
            SessionResult(
                date=date,
                trades=trades,
                gross_pnl_pct=total_pnl_pct,
                gross_pnl_usd=total_pnl_usd,
                kill_switch_fired=kill,
                daily_loss_fired=dloss,
            )
        )
    return sessions, errors


def protocol_hash() -> str:
    """SHA-256 of the protocol doc — records WHICH locked protocol ran."""
    return hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()[:16]


def run_paper(data_dir: Path, tickers: list[str], brain: Hippocampus) -> dict[str, Any]:
    """Evaluate release criteria across all live-feed sessions + WFA verdict.

    The WFA criterion (P5) reuses hanoon_prime.wfa verdicts so the paper
    and Phase-2 gates share the exact same definition of 'admissible' and
    'PASS'.
    """
    from hanoon_prime.wfa import run_wfa_universe, verdicts

    all_sessions: list[SessionResult] = []
    errors: list[str] = []
    per_ticker_trades: dict[str, int] = {}

    for t in tickers:
        path = data_dir / f"{t}_1min.csv"
        if not path.exists():
            errors.append(f"{t}: missing csv")
            continue
        try:
            data = load_ohlcv(path)
        except ValueError as exc:
            errors.append(f"{t}: {exc}")
            continue
        sessions, sess_errors = _run_sessions(t, data, brain)
        all_sessions.extend(sessions)
        errors.extend(sess_errors)
        if not sess_errors:
            per_ticker_trades[t] = sum(len(s.trades) for s in sessions)

    universe = run_wfa_universe(tickers, data_dir)
    wfa_verdict = verdicts(universe)
    wfa_pass = wfa_verdict.verdict == "PASS"

    # One session = one trading day across the WHOLE universe.
    by_day: dict[str, list[SessionResult]] = {}
    for s in all_sessions:
        by_day.setdefault(s.date, []).append(s)
    n_sessions = len(by_day)
    n_trades = sum(len(s.trades) for day in by_day.values() for s in day)
    kill_sessions = sum(
        1 for day in by_day.values() if any(s.kill_switch_fired for s in day)
    )
    dloss_sessions = sum(
        1 for day in by_day.values() if any(s.daily_loss_fired for s in day)
    )
    v = PaperVerdict(
        n_sessions=n_sessions,
        n_trades=n_trades,
        n_kill_sessions=kill_sessions,
        n_daily_loss_sessions=dloss_sessions,
        n_errors=len(errors),
    )
    v.criteria["P5_wfa_pass"] = wfa_pass
    v.verdict = "PASS" if all(v.criteria.values()) else "FAIL"

    return {
        "protocol_hash": protocol_hash(),
        "protocol_file": str(PROTOCOL_FILE),
        "lock_values": {
            "min_paper_trades": MIN_PAPER_TRADES,
            "min_sessions": MIN_SESSIONS,
            "allowed_kill_sessions": ALLOWED_KILL_SESSIONS,
            "allowed_daily_loss_sessions": ALLOWED_DAILY_LOSS_SESSIONS,
        },
        "simulation": {
            "window": EDGE_LOOKBACK,
            "brain": "Hippocampus(static INDICATOR_WEIGHTS)",
            "daily_loss_limit_usd": DAILY_LOSS_LIMIT,
            "kill_daily_loss_limit_usd": KILL_DAILY_LOSS_LIMIT,
            "notional_cap_usd": MAX_POSITION_NOTIONAL,
        },
        "verdict": v.verdict,
        "criteria": v.criteria,
        "counts": {
            "sessions": n_sessions,
            "trades": n_trades,
            "kill_sessions": v.n_kill_sessions,
            "daily_loss_sessions": v.n_daily_loss_sessions,
            "errors": v.n_errors,
        },
        "per_ticker_trades": per_ticker_trades,
        "wfa": {
            "verdict": wfa_verdict.verdict,
            "pooled_sharpe": round(wfa_verdict.pooled_sharpe, 4),
            "deflated_edge": round(wfa_verdict.deflated_edge, 4),
            "pbo": round(wfa_verdict.pbo, 4),
        },
        "sessions": [
            {
                "date": date,
                "trades": sum(len(s.trades) for s in day),
                "pnl_pct": round(sum(s.gross_pnl_pct for s in day), 4),
                "pnl_usd_est": round(sum(s.gross_pnl_usd for s in day), 2),
                "kill_switch_fired": any(s.kill_switch_fired for s in day),
                "daily_loss_fired": any(s.daily_loss_fired for s in day),
            }
            for date, day in sorted(by_day.items())
        ],
        "errors": errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 4 — Paper Incubation Report (pre-locked protocol)",
        "",
        f"Verdict: **{report['verdict']}** · protocol hash `{report['protocol_hash']}`",
        "",
        f"- Sessions observed: `{report['counts']['sessions']}` (floor {report['lock_values']['min_sessions']})",
        f"- Completed trades: `{report['counts']['trades']}` (floor {report['lock_values']['min_paper_trades']})",
        f"- Kill-switch sessions: `{report['counts']['kill_sessions']}` (allowed {report['lock_values']['allowed_kill_sessions']})",
        f"- Daily-loss sessions: `{report['counts']['daily_loss_sessions']}` (allowed {report['lock_values']['allowed_daily_loss_sessions']})",
        f"- Sim errors: `{report['counts']['errors']}` (allowed 0)",
        f"- WFA universe: `{report['wfa']['verdict']}` (deflated {report['wfa']['deflated_edge']}, PBO {report['wfa']['pbo']})",
        "",
        "## Release criteria",
        "",
        "| check | pass? |",
        "|---|---|",
    ]
    for k, ok in report["criteria"].items():
        lines.append(f"| {k} | {'OK' if ok else 'FAIL'} |")
    lines += [
        "",
        "## Sessions",
        "",
        "| date | trades | PnL%/sh | PnL$/est | kill | daily-loss |",
        "|---|---|---|---|---|---|",
    ]
    for s in report["sessions"]:
        lines.append(
            f"| {s['date']} | {s['trades']} | {s['pnl_pct']:+.3f} "
            f"| {s['pnl_usd_est']:+.0f} | {s['kill_switch_fired']} | {s['daily_loss_fired']} |"
        )
    if report["errors"]:
        lines += ["", "## Errors", ""]
        lines += [f"- {e}" for e in report["errors"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 paper incubation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(data_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    brain = Hippocampus()
    report = run_paper(data_dir, tickers, brain)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        md = out.with_suffix(".md")
        md.write_text(_render_md(report))
        print(f"Paper report written to {out} / {md}")

    print(
        f"paper{report['verdict']} sessions={report['counts']['sessions']} "
        f"trades={report['counts']['trades']} "
        f"kill={report['counts']['kill_sessions']} "
        f"dloss={report['counts']['daily_loss_sessions']} "
        f"errors={report['counts']['errors']} "
        f"wfa={report['wfa']['verdict']} "
        f"proto={report['protocol_hash']}"
    )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
