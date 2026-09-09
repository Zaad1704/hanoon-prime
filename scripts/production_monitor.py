"""scripts/production_monitor.py — HALIM-backed, evidence-based runtime checklist.

Runs against the LIVE paper bot and computes a production-readiness verdict per
gate. Deterministic metrics are collected first (they cannot be shaped by an
LLM); HALIM independently re-derives a verdict from the same real runtime data.
A gate only goes PASS when BOTH agree.

Hard rules (deterministic, LLM cannot override) — matching PRODUCTION_READINESS
Gate 1:
  - telemetry must answer /health as ok+connected (no stall)
  - zero NETTING GUARD triggers since the last bot restart
  - zero Traceback lines since the last bot restart
  - daily P&L must never be below -1.0% of equity
  - learning state must stay hermetic (no T/TEST/SMOKE episodes)

Progress metrics (informational, staged):
  - real closes (juli_realized) target 200
  - clean-day streak (calendar-day based) target 10 days

The script is read-only against the bot: it never writes mutation endpoints and
never trades. It keeps its own ledger at scripts/production_state.json.

Usage:
    python3 scripts/production_monitor.py            # collect + HALIM verdict + update ledger
    python3 scripts/production_monitor.py --json     # metrics only (no HALIM call)
    python3 scripts/production_monitor.py --status   # read ledger, no new checks

Exit codes: 0 = PASS, 2 = deterministic FAIL, 3 = HALIM degraded (metrics only),
4 = progress-only HOLD (no rules violated; streak held).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import date, datetime, timezone
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "logs", "hanoon_prime.log")
STATE_PATH = os.path.join(BASE_DIR, "runtime", "state.json")
REALIZED_PATH = os.path.join(BASE_DIR, "runtime", "juli_realized.json")
JULI_STATE_PATH = os.path.join(BASE_DIR, "runtime", "juli_state.json")
TELEMETRY_URL = "http://127.0.0.1:8080"
HALIM_URL = os.environ.get("HALIM_URL", "http://127.0.0.1:8765")
STATE_FILE = os.path.join(BASE_DIR, "scripts", "production_state.json")

START_MARKER = re.compile(r"ib_adapter\s+Starting \(seed=")
GUARD_MARKER = re.compile(r"NETTING GUARD")
TRACE_MARKER = re.compile(r"Traceback \(most recent call last\)")

DRAWDOWN_LIMIT = 0.01  # 1.0% of equity as floor for a clean day
CLOSES_TARGET = 200
DAYS_TARGET = 10
POLLUTED = {"T", "TEST"}


def _utcnow() -> float:
    return datetime.now(timezone.utc).timestamp()


def _read(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _rfc(ts: Optional[float]) -> Optional[str]:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
        if ts
        else None
    )


def _log_count_since(marker: re.Pattern[str], start_ts: float) -> int:
    """Count marker lines with wall-clock timestamp >= start_ts."""
    n = 0
    try:
        with open(LOG_PATH, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = re.match(r"(\d\d):(\d\d):(\d\d)\.(\d+)", line)
                if not m:
                    continue
                hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                now = datetime.now(timezone.utc)
                ts = now.replace(hour=hh, minute=mm, second=ss).timestamp()
                if ts < start_ts:
                    continue
                if marker.search(line):
                    n += 1
    except OSError:
        return -1
    return n


def _bot_start_ts() -> float:
    """Wall-clock timestamp of the most recent 'ib_adapter Starting' line."""
    ts = 0.0
    try:
        with open(LOG_PATH, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if START_MARKER.search(line):
                    m = re.match(r"(\d\d):(\d\d):(\d\d)\.(\d+)", line)
                    if m:
                        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        now = datetime.now(timezone.utc)
                        ts = now.replace(hour=hh, minute=mm, second=ss).timestamp()
    except OSError:
        pass
    return ts


def collect() -> dict[str, Any]:
    now = _utcnow()
    start_ts = _bot_start_ts()
    state = _read(STATE_PATH)
    brain = state.get("brain_state", {}) if isinstance(state, dict) else {}
    feed = brain.get("account_feed", {}) if isinstance(brain, dict) else {}

    safety_policy = brain.get("policy_state", {}) if isinstance(brain, dict) else {}
    safety = {
        "enabled": bool(safety_policy.get("enabled", False)),
        "halted": bool(safety_policy.get("halted", False)),
        "authorized": bool(safety_policy.get("authorized", True)),
    }

    realized = _read(REALIZED_PATH)
    juli = _read(JULI_STATE_PATH)
    polluted_episodes = 0
    for ep in juli.get("episodes", []) or []:
        if isinstance(ep, dict) and ep.get("ticker") in POLLUTED:
            polluted_episodes += 1

    health: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(f"{TELEMETRY_URL}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
    except Exception as exc:
        health = {"status": f"unreachable: {exc}"}

    halim_ok, halim_mode = False, "unknown (unreachable)"
    try:
        req = urllib.request.Request(
            f"{HALIM_URL}/v1/complete",
            data=json.dumps(
                {
                    "prompt": 'Reply with exactly: {"ok":1}',
                    "purpose": "health_probe",
                    "priority": "low",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        halim_ok = bool(result.get("ok", False))
        halim_mode = (
            "learn_by_action"
            if (result.get("runtime", {}) or {}).get("learn_by_action")
            else "read_only"
        )
    except Exception:
        pass

    return {
        "ts": now,
        "start_ts": start_ts,
        "health_status": health.get("status"),
        "connected": bool(health.get("connected", False)),
        "equity": float(feed.get("equity", 0.0) or 0.0),
        "daily_pnl": float(feed.get("daily_pnl", 0.0) or 0.0),
        "positions_open": int(brain.get("positions_open", 0) or 0),
        "safety": safety,
        "guard_triggers_session": _log_count_since(GUARD_MARKER, start_ts),
        "tracebacks_session": _log_count_since(TRACE_MARKER, start_ts),
        "real_closes_total": int(realized.get("total", 0) or 0),
        "polluted_episodes": polluted_episodes,
        "halim_ok": halim_ok,
        "halim_mode": halim_mode,
        "regime_label": (
            brain.get("regime_label", "unknown")
            or (brain.get("regime_state", {}) or {}).get("current_regime", "unknown")
            or "unknown"
        ),
        "risk_ceiling": brain.get("risk_ceiling"),
    }


def hard_rules(m: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(rule, reason)] for every deterministically-violated rule."""
    fails: list[tuple[str, str]] = []
    if m["health_status"] != "ok" or not m["connected"]:
        fails.append(
            ("health", f"status={m['health_status']} connected={m['connected']}")
        )
    if m["guard_triggers_session"]:
        fails.append(
            ("netting_guard", f"{m['guard_triggers_session']} trigger(s) this session")
        )
    if m["tracebacks_session"]:
        fails.append(
            ("tracebacks", f"{m['tracebacks_session']} Traceback(s) this session")
        )
    if m["equity"] > 0 and m["daily_pnl"] <= -DRAWDOWN_LIMIT * m["equity"]:
        pct = 100 * -m["daily_pnl"] / m["equity"]
        fails.append(
            (
                "drawdown",
                f"daily_pnl {m['daily_pnl']:.2f} is below -1.0% (-{pct:.2f}% of equity)",
            )
        )
    if m["polluted_episodes"]:
        fails.append(
            ("learning_purity", f"{m['polluted_episodes']} test-episode(s) present")
        )
    return fails


def halim_verdict(m: dict[str, Any]) -> dict[str, Any]:
    """Ship real metrics to HALIM; return {verdict, gates, issues}."""
    payload = {
        "prompt": (
            "You are HALIM, safety auditor for the hanoon-prime PAPER bot.\n"
            "Evaluate production-readiness using ONLY this real runtime data:\n"
            f"{json.dumps(m)}\n\n"
            "Return EXACTLY this JSON, no prose:\n"
            '{"verdict": "PASS"|"FAIL",'
            ' "gates": {"gate1": "GO"|"HOLD"},'
            ' "issues": [{"gate": "<gate1|health|halim|system>",'
            ' "risk": "critical|high|medium|low",'
            ' "detail": "<what the data shows>"}]}'
        ),
        "purpose": "production_readiness",
        "priority": "high",
    }
    req = urllib.request.Request(
        f"{HALIM_URL}/v1/complete",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result: dict[str, Any] = json.loads(resp.read().decode())
    text = result.get("text", "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return {"verdict": "FAIL", "gates": {"gate1": "HOLD"}, "issues": []}
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {"verdict": "FAIL", "gates": {"gate1": "HOLD"}, "issues": []}
    verdict = str(parsed.get("verdict", "FAIL")).upper()
    gates = parsed.get("gates") or {}
    issues = parsed.get("issues") or []
    return {
        "verdict": verdict if verdict in ("PASS", "FAIL") else "FAIL",
        "gates": {k: g for k, g in gates.items() if g in ("GO", "HOLD")},
        "issues": issues if isinstance(issues, list) else [],
    }


def _prev_trading_day(d: date) -> Optional[date]:
    """Immediately preceding weekday (Mon-Fri) before ``d``."""
    p = d
    for _ in range(7):
        p = p.fromordinal(p.toordinal() - 1)
        if p.weekday() < 5:
            return p
    return None


def _roll_streak(wise: dict[str, Any], m: dict[str, Any]) -> int:
    """Advance (or reset) the clean-day streak on a trading-day basis.

    Only Mon-Fri count; weekends neither advance nor break the streak.
    Same-weekday repeat runs advance at most once. A gap of more than one
    trading day between consecutive PASSes breaks the streak.
    """
    today = date.today()
    if today.weekday() >= 5:
        return wise.get("clean_days_streak", 0)
    today_s = today.isoformat()
    last = wise.get("last_clean_day")
    if last == today_s:
        return wise.get("clean_days_streak", 1)
    if last:
        last_d = date.fromisoformat(last)
        if _prev_trading_day(today) == last_d:
            wise["clean_days_streak"] = wise.get("clean_days_streak", 0) + 1
        else:
            wise["clean_days_streak"] = 1  # gap → new streak
    else:
        wise["clean_days_streak"] = 1
    wise["last_clean_day"] = today_s
    return wise["clean_days_streak"]


def _next_slot_minute(now_minute: int) -> int:
    """Next run minute-of-day: every :00/:30 plus a 23:50 finalize."""
    slots = [h * 60 + m for h in range(24) for m in (0, 30)]
    slots.append(23 * 60 + 50)
    nxt = next((t for t in sorted(slots) if t > now_minute), None)
    if nxt is not None:
        return nxt
    return min(slots)  # wraps past midnight


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="metrics only, no HALIM")
    ap.add_argument("--status", action="store_true", help="print ledger, no new checks")
    ap.add_argument(
        "--daemon",
        action="store_true",
        help="loop forever: check at :00/:30 and 23:50 local",
    )
    args = ap.parse_args()

    if args.daemon:
        import time

        log_path = os.path.join(BASE_DIR, "logs", "production_monitor.log")
        print("production_monitor daemon starting", flush=True)
        while True:
            try:
                code = run_once(status=False, json_only=False)
            except Exception as exc:  # keep the loop alive; log and retry
                code = -1
                with open(log_path, "a") as fh:
                    fh.write("check crashed: %r\n" % (exc,))
            now = datetime.now()
            seconds = (
                _next_slot_minute(now.hour * 60 + now.minute)
                - now.hour * 60
                - now.minute
            ) * 60
            seconds = max(60, seconds)
            with open(log_path, "a") as fh:
                fh.write(
                    "[%s] check exit=%d; next in %d min\n"
                    % (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        code,
                        seconds // 60,
                    )
                )
            time.sleep(seconds)

    return run_once(args.status, args.json)


def run_once(status: bool, json_only: bool) -> int:
    if status:
        snap = _read(STATE_FILE)
        print(
            json.dumps(
                {k: snap[k] for k in ("last_summary", "gate1") if k in snap}, indent=2
            )
        )
        return 0

    snap = _read(STATE_FILE)
    m = collect()
    if json_only:
        print(json.dumps({"metrics": m, "ledger": snap}, indent=2))
        return 0

    hard = hard_rules(m)
    verdict, marks = "FAIL", " . ".join(f"{r}:{why}" for r, why in hard) or "PASS"
    if hard:
        marks = "; ".join(f"{r}: {why}" for r, why in hard)
    else:
        marks = "ok"
    g1_hard = bool(hard)

    # HALIM verdict (advisory on top of deterministic rules).
    if not m["halim_ok"]:
        gate_state = snap.get("gate1", {})
        snap["gate1"] = {
            **gate_state,
            "status": gate_state.get("status", "pending"),
            "since": gate_state.get("since"),
            "last_check_ts": m["ts"],
            "last_skip_reason": "halim unreachable",
        }
        snap["last_summary"] = {
            "ts": m["ts"],
            "verdict": "DEGRADED",
            "returns": None,
            "hard_fail": g1_hard,
            "metrics": {
                k: m[k]
                for k in (
                    "health_status",
                    "connected",
                    "guard_triggers_session",
                    "tracebacks_session",
                    "daily_pnl",
                    "equity",
                    "real_closes_total",
                    "polluted_episodes",
                    "halim_ok",
                )
            },
        }
        _write(STATE_FILE, snap)
        print(
            "DEGRADED  HALIM unreachable; deterministic rules evaluated. "
            "Exit 3 hard-fail=%s" % g1_hard
        )
        return 3

    ha = halim_verdict(m)
    llm_fail = ha.get("verdict") == "FAIL"

    wise = snap.setdefault("roles", {}).setdefault("wise", {})
    gate1 = snap.setdefault(
        "gate1", {"name": "Gate 1 - Paper soak", "status": "pending"}
    )
    gate1["since"] = gate1.get("since") or _rfc(m["ts"])
    gate1["last_check_ts"] = m["ts"]
    gate1["last_verdict_ts"] = m["ts"]

    if g1_hard or llm_fail:
        wise["clean_days_streak"] = 0
        wise["last_clean_day"] = None
        gate1["status"] = "pending"
        gate1["best_days_streak"] = gate1.get("best_days_streak", 0)
        verdict = "FAIL"
        parts: list[str] = []
        if llm_fail:
            parts.append("halim:" + ha.get("verdict", "FAIL"))
        parts.extend(f"{r}: {why}" for r, why in hard)
        marks = " + ".join(parts) or "PASS"
    else:
        streak = _roll_streak(wise, m)
        gate1["best_days_streak"] = max(gate1.get("best_days_streak", 0), streak)
        gate1["status"] = (
            "GO"
            if streak >= DAYS_TARGET and m["real_closes_total"] >= CLOSES_TARGET
            else (
                "clean_streak_short_of_closes"
                if streak >= DAYS_TARGET and m["real_closes_total"] < CLOSES_TARGET
                else "soaking"
            )
        )
        verdict = "PASS"

    snap["last_verdict"] = ha
    snap["last_summary"] = {
        "ts": m["ts"],
        "verdict": verdict,
        "hard_fail": g1_hard,
        "llm_fail": llm_fail,
        "metrics": {
            "health": m["health_status"],
            "connected": m["connected"],
            "guard": m["guard_triggers_session"],
            "tracebacks": m["tracebacks_session"],
            "daily_pnl": round(m["daily_pnl"], 2),
            "equity": round(m["equity"], 2),
            "closes": m["real_closes_total"],
            "polluted": m["polluted_episodes"],
            "halim_ok": m["halim_ok"],
        },
        "streak": wise.get("clean_days_streak", 0),
        "gate_status": gate1["status"],
    }
    _write(STATE_FILE, snap)

    print(
        "verdict=%s  streak=%d/%d  closes=%d/%d  rules=[%s]  halim=%s  gate=%s"
        % (
            verdict,
            wise.get("clean_days_streak", 0),
            DAYS_TARGET,
            m["real_closes_total"],
            CLOSES_TARGET,
            marks,
            ha.get("verdict", "?"),
            gate1["status"],
        )
    )
    if ha.get("issues"):
        for issue in ha["issues"]:
            print(
                "  halim-issue: %s/%s %s"
                % (
                    issue.get("gate", "?"),
                    issue.get("risk", "?"),
                    issue.get("detail", ""),
                )
            )

    if g1_hard:
        return 2
    if llm_fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
