"""scripts/production_monitor.py — full-pipeline guardian + bug catcher.

Runs against the LIVE bot and produces a production-readiness verdict for the
Gate-1 soak, PLUS a bug-catching pass that diagnoses and notifies on any
anomaly. Two roles, one process:

ROLE 1 — VERIFY (deterministic, HALIM cannot override):
  - telemetry /health must be ok+connected and /snapshot non-empty
  - pipeline must be alive: a fresh CYCLE line within 5 min
  - zero NETTING GUARD triggers since the last bot restart
  - zero Traceback lines since the last bot restart
  - daily P&L must never break -1.0% of equity
  - learning state must stay hermetically test-free
  Any violation is a FAIL (exit 2) and resets the clean-day streak.

ROLE 2 — CATCH BUGS (anomalies; do NOT reset the streak):
  - decision-path sanity: finite scores in the last N journal verdicts,
    valid verdict actions, weighted carriers sane, brain-state fields typed
    and bounded
  - operational noise funnel: ERROR lines and pipeline_incidents since start,
    minus a known-bug allowlist
  - halt/block markers (SAFETY HALT, LEARN BLOCKED)
  Anomalies go to HALIM, which emits a bug report (root-cause hypothesis,
  severity, suggested fix); the user is notified once per signature per day
  via Telegram (reusing hanoon_prime._telegram). If Telegram is not
  configured, findings are recorded in the ledger instead.

LIFECYCLE: run by scripts/start.command alongside the bot; stop.command tears
it down. When the bot is unreachable the daemon idles (no FAIL flood, no
notify) and exits gracefully after ~4h of sustained downtime.

Usage:
    python3 scripts/production_monitor.py            # single verify+bug-catch pass
    python3 scripts/production_monitor.py --json     # metrics only
    python3 scripts/production_monitor.py --status   # read ledger
    python3 scripts/production_monitor.py --daemon   # loop; used by start.command

Exit codes: 0 PASS · 2 deterministic FAIL · 3 HALIM degraded · 4 anomalies found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)  # allow reuse of hanoon_prime._telegram

LOG_PATH = os.path.join(BASE_DIR, "logs", "hanoon_prime.log")
STATE_PATH = os.path.join(BASE_DIR, "runtime", "state.json")
REALIZED_PATH = os.path.join(BASE_DIR, "runtime", "juli_realized.json")
JULI_STATE_PATH = os.path.join(BASE_DIR, "runtime", "juli_state.json")
JOURNAL_PATH = os.path.join(BASE_DIR, "runtime", "journal_live.jsonl")
TELEMETRY_URL = "http://127.0.0.1:8080"
HALIM_URL = os.environ.get("HALIM_URL", "http://127.0.0.1:8765")
STATE_FILE = os.path.join(BASE_DIR, "scripts", "production_state.json")

START_MARKER = re.compile(r"ib_adapter\s+Starting \(seed=")
GUARD_MARKER = re.compile(r"NETTING GUARD")
TRACE_MARKER = re.compile(r"Traceback \(most recent call last\)")
CYCLE_MARKER = re.compile(r"ib_cycle\s+CYCLE ")
HEARTBEAT_MARKER = re.compile(r"ib_cycle\s+HEARTBEAT")
SUB_MARKER = re.compile(r"ib_streamer\s+Subscribed to (\S+)")
ERROR_MARKER = re.compile(r"\bERROR\b")
SNAPSHOT_FAIL_MARKER = re.compile(r"telemetry\s+snapshot build failed")
SAFETY_HALT_MARKER = re.compile(r"SAFETY HALT")
LEARN_BLOCKED_MARKER = re.compile(r"LEARN BLOCKED")

POLLUTED = {"T", "TEST"}
VALID_ACTIONS = {"BUY", "SELL", "HOLD", "VETOED", "PASS", "OPEN", "CLOSE"}
NOISE_ERROR = re.compile(
    r"cancelMktData|latency spike|Max retries exceeded|ib_insync|"
    r"TWS error 1101|EClient error|Connection reset|BadMessage"
)
DRAWDOWN_LIMIT = 0.01
CLOSES_TARGET = 200
DAYS_TARGET = 10
CYCLE_STALE_SEC = 300  # a live cycle every ~3s; 5 min silence = pipeline stall
VERDICT_SAMPLE = 60
BOT_DOWN_EXIT_CYCLES = 8  # ~4h of sustained downtime → self-exit


def _logfile() -> str:
    return os.path.join(BASE_DIR, "logs", "production_monitor.log")


def _note(msg: str) -> None:
    with open(_logfile(), "a") as fh:
        fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def _read(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _ts_of(line: str) -> Optional[float]:
    m = re.match(r"(\d\d):(\d\d):(\d\d)\.\d+", line)
    if not m:
        return None
    # Log stamps are LOCAL wall-clock (hh:mm:ss, no date). Build an absolute
    # timestamp from the local 'now' with the log's time substituted — the
    # naive-local timestamp() matches the same clock the bot logs from.
    now = datetime.now()
    try:
        return now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3))
        ).timestamp()
    except ValueError:
        return None


def _bot_start_lines() -> list[str]:
    """Every log line after the LAST 'ib_adapter Starting' marker.

    The log stamps HH:MM:SS only (no date), so per-line timestamp math cannot
    tell this session from yesterday's. Session bounding by marker position is
    timezone/day-proof: everything before the last start marker is ignored.
    """
    session: list[str] = []
    try:
        with open(LOG_PATH, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if START_MARKER.search(line):
                    session = []
                else:
                    session.append(line)
    except OSError:
        return []
    return session


def _count_in(lines: list[str], marker: re.Pattern[str]) -> int:
    return sum(1 for line in lines if marker.search(line))


def _last_line_age(marker: re.Pattern[str]) -> Optional[float]:
    """Age in seconds of the most recent matching log line (None if absent)."""
    latest = None
    try:
        with open(LOG_PATH, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if marker.search(line):
                    t = _ts_of(line)
                    if t is not None:
                        latest = t
    except OSError:
        return None
    return (time.time() - latest) if latest is not None else None


def _tail(path: str, nbytes: int = 131072) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            return fh.read().decode(errors="ignore")
    except OSError:
        return ""


def _journal_sample() -> dict[str, Any]:
    """Tail the most recent journal window: verdict validity + incidents."""
    tail = _tail(JOURNAL_PATH)
    rows, incidents = [], 0
    for line in tail.splitlines():
        if '"event": "pipeline_incident"' in line:
            incidents += 1
            continue
        if '"event": "verdict"' not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows.append(r)
    sample = rows[-VERDICT_SAMPLE:]
    bad_action, nan_score = 0, 0
    for r in sample:
        if str(r.get("action", "")).upper() not in VALID_ACTIONS:
            bad_action += 1
        sc = r.get("score")
        if sc is None or not isinstance(sc, (int, float)) or sc != sc:  # NaN check
            nan_score += 1
    return {
        "sample": len(sample),
        "invalid_actions": bad_action,
        "nan_scores": nan_score,
        "incidents": incidents,
        "last_action": str(sample[-1].get("action", "")) if sample else "",
        "last_ticker": str(sample[-1].get("ticker", "")) if sample else "",
        "last_stage": str(sample[-1].get("stage", "")) if sample else "",
    }


def _log_errors_in(lines: list[str]) -> dict[str, Any]:
    """ERROR lines + telemetry snapshot rebuild failures in this session."""
    total, flagged, samples = 0, 0, []
    snapshot_fails = 0
    for line in lines:
        if SNAPSHOT_FAIL_MARKER.search(line):
            snapshot_fails += 1
            continue
        if ERROR_MARKER.search(line):
            total += 1
            if NOISE_ERROR.search(line):
                continue
            flagged += 1
            if len(samples) < 8:
                samples.append(line.strip()[:200])
    return {
        "errors_total": total,
        "problems": flagged,
        "samples": samples,
        "snapshot_fails": snapshot_fails,
    }


def _collect() -> dict[str, Any]:
    state = _read(STATE_PATH)
    brain = state.get("brain_state", {})
    session = _bot_start_lines()

    # health + snapshot (telemetry stall detection)
    health: dict[str, Any] = {}
    snapshot: dict[str, Any] = {}
    for url, out in (
        (f"{TELEMETRY_URL}/health", health),
        (f"{TELEMETRY_URL}/snapshot", snapshot),
    ):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                out.update(json.loads(resp.read().decode()))
        except Exception as exc:
            out["_unreachable"] = str(exc)
    snapshot_ok = (
        bool(snapshot)
        and isinstance(snapshot.get("health"), dict)
        and not snapshot.get("_unreachable")
    )
    health_ok = health.get("status") == "ok" and bool(health.get("connected", False))

    # brain-state field sanity
    threshold = brain.get("threshold")
    pred_error = brain.get("pred_error")
    risk_ceiling = brain.get("risk_ceiling")
    regime = brain.get("regime_risk") or (brain.get("regime_state") or {}).get(
        "current_regime"
    )
    pos = brain.get("positions_open", 0)
    policy = brain.get("policy_state", {})
    field_issues = []
    if not (isinstance(threshold, (int, float)) and 0.45 <= threshold <= 0.70):
        field_issues.append(f"threshold={threshold!r} out of [0.45,0.70]")
    # pred_error is legitimately None in normal operation; only flag a numeric
    # value that has drifted outside the model's output range.
    if pred_error is not None and not (
        isinstance(pred_error, (int, float)) and 0.0 <= pred_error <= 1.0
    ):
        field_issues.append(f"pred_error={pred_error!r} not in [0,1]")
    if not (isinstance(risk_ceiling, (int, float)) and risk_ceiling > 0):
        field_issues.append(f"risk_ceiling={risk_ceiling!r} not positive")
    if not isinstance(pos, int) or pos < 0:
        field_issues.append(f"positions_open={pos!r} not a non-negative int")

    # weights sanity (juli_state); sum |w| within enforcer band
    juli = _read(JULI_STATE_PATH)
    weights = juli.get("weights", {}) if isinstance(juli, dict) else {}
    w_vals = [v for v in weights.values() if isinstance(v, (int, float))]
    weights_issue = None if w_vals else "weights empty"
    if w_vals and not weights_issue:
        if any(v != v or v in (float("inf"), float("-inf")) for v in w_vals):
            weights_issue = "weights contain NaN/inf"
        elif not (all(-2.0 <= v <= 2.0 for v in w_vals)):
            weights_issue = "weights outside [-2,2]"
        elif len(w_vals) < 10:
            weights_issue = "weights unexpectedly sparse"

    # realized + episodes sanity
    realized = _read(REALIZED_PATH)
    total_closes = int(realized.get("total", 0) or 0)
    polluted_episodes = sum(
        1
        for ep in (juli.get("episodes", []) or [])
        if isinstance(ep, dict) and ep.get("ticker") in POLLUTED
    )

    journal = _journal_sample()
    log_err = _log_errors_in(session)
    settled = {
        "guard": _count_in(session, GUARD_MARKER),
        "tracebacks": _count_in(session, TRACE_MARKER),
        "safety_halt": _count_in(session, SAFETY_HALT_MARKER),
        "learn_blocked": _count_in(session, LEARN_BLOCKED_MARKER),
    }
    # Liveness is gauged on HEARTBEAT (logged every 60s in every session
    # state) rather than CYCLE (logged only while the session is active) —
    # an overnight-sleeping bot must not look "stalled".
    heartbeat_age = _last_line_age(HEARTBEAT_MARKER)
    cycle_age = _last_line_age(CYCLE_MARKER)

    # subscriptions present in session?
    subs = set(SUB_MARKER.findall(_tail(LOG_PATH, 1 << 20)))
    feed = brain.get("account_feed", {})
    return {
        "ts": time.time(),
        "session_lines": len(session),
        "health_status": health.get("status"),
        "connected": health.get("connected", False),
        "snapshot_ok": snapshot_ok,
        "heartbeat_age_s": heartbeat_age,
        "cycle_age_s": cycle_age,
        "session_active": bool(health.get("session_active", False)),
        "equity": float(feed.get("equity", 0.0) or 0.0),
        "daily_pnl": float(feed.get("daily_pnl", 0.0) or 0.0),
        "positions_open": pos,
        "regime": regime,
        "threshold": threshold,
        "pred_error": pred_error,
        "risk_ceiling": risk_ceiling,
        "safety": {
            "enabled": bool(policy.get("enabled", False)),
            "halted": bool(policy.get("halted", False)),
            "authorized": bool(policy.get("authorized", True)),
        },
        "field_issues": field_issues,
        "weights_issue": weights_issue,
        "journal": journal,
        "errors": log_err,
        "incidents": journal["incidents"],
        "counters": settled,
        "subscriptions": len(subs),
        "real_closes_total": total_closes,
        "polluted_episodes": polluted_episodes,
    }


def _hard_rules(m: dict[str, Any]) -> list[tuple[str, str]]:
    fails = []
    if not m["health_ok"]:
        fails.append(
            ("health", f"status={m['health_status']} connected={m['connected']}")
        )
    if not m["snapshot_ok"]:
        fails.append(("telemetry_stall", "snapshot empty/unreachable"))
    if m["heartbeat_age_s"] is not None and m["heartbeat_age_s"] > CYCLE_STALE_SEC:
        fails.append(
            ("pipeline_stall", f"last HEARTBEAT {m['heartbeat_age_s']:.0f}s ago")
        )
    if m["counters"]["guard"]:
        fails.append(("netting_guard", f"{m['counters']['guard']} trigger(s)"))
    if m["counters"]["tracebacks"]:
        fails.append(("tracebacks", f"{m['counters']['tracebacks']} Traceback(s)"))
    if m["equity"] > 0 and m["daily_pnl"] <= -DRAWDOWN_LIMIT * m["equity"]:
        pct = 100 * -m["daily_pnl"] / m["equity"]
        fails.append(
            ("drawdown", f"{m['daily_pnl']:.2f} = -{pct:.2f}% of equity (floor -1.0%)")
        )
    if m["polluted_episodes"]:
        fails.append(("learning_purity", f"{m['polluted_episodes']} test episode(s)"))
    return fails


def _anomalies(m: dict[str, Any]) -> list[tuple[str, str]]:
    """Non-streak-resetting findings; the bug-catcher feed."""
    out = []
    if m["field_issues"]:
        out.append(("brain_state", "; ".join(m["field_issues"])))
    if m["weights_issue"]:
        out.append(("weights", m["weights_issue"]))
    if m["journal"]["nan_scores"]:
        out.append(
            ("scores_nan", f"{m['journal']['nan_scores']}/{m['journal']['sample']} NaN")
        )
    if m["journal"]["invalid_actions"]:
        out.append(("verdict_actions", f"{m['journal']['invalid_actions']} invalid"))
    if m["errors"]["problems"]:
        out.append(("log_errors", f"{m['errors']['problems']} non-noise ERROR lines"))
    if m["errors"]["snapshot_fails"]:
        out.append(
            (
                "telemetry_snapshot",
                f"{m['errors']['snapshot_fails']} failed snapshot rebuild(s)",
            )
        )
    if m["incidents"]:
        out.append(("journal_incidents", f"{m['incidents']} pipeline_incident(s)"))
    if m["counters"]["safety_halt"]:
        out.append(("safety_halt", f"{m['counters']['safety_halt']} trip(s)"))
    if m["counters"]["learn_blocked"]:
        out.append(
            ("learn_blocked", f"{m['counters']['learn_blocked']} blocked close(s)")
        )
    return out


def _halim_call_prompt(m: dict[str, Any]) -> str:
    return (
        "You are HALIM, safety auditor AND bug catcher for the hanoon-prime "
        "PAPER bot. Using ONLY this real runtime data:\n"
        f"{json.dumps(m, default=str)}\n\n"
        "Return EXACTLY this JSON, no prose:\n"
        '{"verdict": "PASS"|"FAIL", "gates": {"gate1": "GO"|"HOLD"}, '
        '"issues": [{"gate": "<gate1|health|pipeline|halim|system>", '
        '"risk": "critical|high|medium|low", '
        '"detail": "<what the data shows>"}], '
        '"bug_report": {"found": true|false, "summary": "<what broke>", '
        '"root_cause": "<hypothesis>", "severity": "critical|high|medium|low", '
        '"suggested_fix": "<one-line actionable fix>"}}'
    )


def _halim_call(metrics: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{HALIM_URL}/v1/complete",
        data=json.dumps(
            {
                "prompt": _halim_call_prompt(metrics),
                "purpose": "production_readiness",
                "priority": "high",
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result: dict[str, Any] = json.loads(resp.read().decode())
    text = result.get("text", "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    verdict = str(parsed.get("verdict", "FAIL")).upper()
    gates = parsed.get("gates") or {}
    issues = parsed.get("issues") or []
    bug = parsed.get("bug_report") or {}
    return {
        "verdict": verdict if verdict in ("PASS", "FAIL") else "FAIL",
        "gates": {k: g for k, g in gates.items() if g in ("GO", "HOLD")},
        "issues": issues if isinstance(issues, list) else [],
        "bug_report": bug if isinstance(bug, dict) else {},
    }


def _halim_probe() -> str:
    """Return "ok", "asleep" (expected post-market), or "down" (degraded)."""
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
            res = json.loads(resp.read().decode())
        if res.get("ok"):
            return "ok"
        if res.get("reason") == "system_asleep":
            return "asleep"
        return "down"
    except urllib.error.HTTPError as exc:
        # A sleeping HALIM rejects inference with 503 +
        # {"ok": false, "reason": "system_asleep"} — read the body.
        try:
            res = json.loads(exc.read().decode(errors="replace"))
        except (json.JSONDecodeError, ValueError):
            return "down"
        if res.get("reason") == "system_asleep":
            return "asleep"
        return "down"
    except Exception:
        return "down"


def _notify(text: str) -> bool:
    try:
        from hanoon_prime._telegram import send  # reused bot notify path

        return bool(send(text))
    except Exception:
        return False


def _alert(day: str, key: str, text: str, ledger: dict[str, Any]) -> None:
    """Deduped notify: at most one Telegram per signature per day."""
    alerted = ledger.setdefault("alerted", {}).setdefault(day, {})
    if alerted.get(key):
        return
    ok = _notify(text)
    if not ok:
        _note(f"notify skipped (telegram unconfigured/rate-limited): {key}")
    alerted[key] = {"ts": time.time(), "sent": ok}
    _write(STATE_FILE, ledger)


def _prev_trading_day(d: date) -> Optional[date]:
    for _ in range(10):
        d = d.fromordinal(d.toordinal() - 1)
        if d.weekday() < 5:
            return d
    return None


def _roll_streak(wise: dict[str, Any], today: date) -> int:
    if today.weekday() >= 5:
        return wise.get("clean_days_streak", 0)
    today_s = today.isoformat()
    last = wise.get("last_clean_day")
    if last == today_s:
        return wise.get("clean_days_streak", 1)
    prev = _prev_trading_day(today)
    if last and prev and last == prev.isoformat():
        wise["clean_days_streak"] = wise.get("clean_days_streak", 0) + 1
    else:
        wise["clean_days_streak"] = 1
    wise["last_clean_day"] = today_s
    return wise["clean_days_streak"]


def run_once(today_s: str, ledger: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    m = _collect()
    m["health_ok"] = m["health_status"] == "ok" and bool(m["connected"])

    hard = _hard_rules(m)
    anomalies = _anomalies(m)

    # HALIM adjudicates only when awake; a post-market 'asleep' HALIM is
    # expected (deterministic verdict stands), 'down' is genuine degradation.
    halim_state = _halim_probe()
    verdict = "FAIL" if hard else ("PASS")

    ha: dict[str, Any] = {}
    if halim_state == "ok":
        ha = _halim_call(m)
        verdict = "FAIL" if (hard or ha.get("verdict") == "FAIL") else verdict
    elif halim_state == "down":
        verdict = verdict  # deterministic verdict stands; exit 3 below

    wise = ledger.setdefault("roles", {}).setdefault("wise", {})
    gate1 = ledger.setdefault(
        "gate1", {"name": "Gate 1 - Paper soak", "status": "pending"}
    )
    gate1["since"] = gate1.get("since") or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    if hard:
        wise["clean_days_streak"] = 0
        wise["last_clean_day"] = None
        gate1["status"] = "pending"
    else:
        streak = _roll_streak(wise, date.today())
        gate1["best_days_streak"] = max(gate1.get("best_days_streak", 0), streak)
        gate1["status"] = (
            "GO"
            if streak >= DAYS_TARGET and m["real_closes_total"] >= CLOSES_TARGET
            else ("soaking" if streak < DAYS_TARGET else "clean_but_short_on_closes")
        )

    # ── bug-catcher: HALIM diagnosis + deduped notify per signature/day ──
    bug = ha.get("bug_report") or {}
    if anomalies and not bug.get("found"):
        # HALIM silent/degraded? fall back to a deterministic summary — the
        # user must still be told, so notify on the raw findings.
        bug = {
            "found": True,
            "summary": "; ".join(f"{r}: {d}" for r, d in anomalies),
            "root_cause": "deterministic anomaly feed (halim silent)",
            "severity": "medium",
            "suggested_fix": "inspect ledger / logs/production_monitor.log",
        }
    for rule, detail in anomalies:
        _alert(
            today_s,
            rule,
            (
                "HANOON BUG-CATCHER\n"
                f"finding: {rule} — {detail}\n"
                f"halim diagnosis: {bug.get('summary', 'n/a')}\n"
                f"root cause: {bug.get('root_cause', 'n/a')}\n"
                f"severity: {bug.get('severity', 'medium')}\n"
                f"suggested fix: {bug.get('suggested_fix', 'n/a')}"
            ),
            ledger,
        )
        ledger.setdefault("findings", {}).setdefault(today_s, {})[rule] = {
            "detail": detail,
            "ts": time.time(),
            "halim": bool(bug.get("found")),
        }

    ledger["last_verdict_ts"] = m["ts"]
    ledger["last_summary"] = {
        "ts": m["ts"],
        "verdict": verdict,
        "hard_fail": bool(hard),
        "anomalies": [f"{r}: {d}" for r, d in anomalies],
        "metrics": {
            "health": m["health_status"],
            "snapshot_ok": m["snapshot_ok"],
            "heartbeat_age_s": round(m["heartbeat_age_s"] or 0, 1),
            "cycle_age_s": round(m["cycle_age_s"] or 0, 1),
            "guard": m["counters"]["guard"],
            "tracebacks": m["counters"]["tracebacks"],
            "daily_pnl": round(m["daily_pnl"], 2),
            "equity": round(m["equity"], 2),
            "closes": m["real_closes_total"],
            "polluted": m["polluted_episodes"],
            "halim_state": halim_state,
            "subscriptions": m["subscriptions"],
        },
        "streak": wise.get("clean_days_streak", 0),
        "gate_status": gate1["status"],
        "bug_report": bug,
    }
    _write(STATE_FILE, ledger)
    if hard:
        code = 2
    elif halim_state == "down":
        code = 3  # HALIM degraded — deterministic checks above still ran
    elif anomalies:
        code = 4  # bugs found and reported to the user
    else:
        code = 0
    return code, ledger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    args = ap.parse_args()

    if args.json:
        print(json.dumps(_collect(), indent=2, default=str))
        return 0
    if args.status:
        led = _read(STATE_FILE)
        print(
            json.dumps(
                {
                    k: led.get(k)
                    for k in ("last_summary", "gate1", "findings", "alerted")
                    if k in led
                },
                indent=2,
            )
        )
        return 0
    if args.daemon:
        _note("production_monitor daemon starting")
        down_cycles = 0
        while True:
            ledger = _read(STATE_FILE)
            code, _led = run_once(date.today().isoformat(), ledger)
            try:
                with urllib.request.urlopen(
                    f"{TELEMETRY_URL}/health", timeout=5
                ) as resp:
                    json.loads(resp.read().decode())
                down_cycles = 0
            except Exception:
                down_cycles += 1
                _note(f"bot unreachable (cycle {down_cycles}); idling")
                if down_cycles >= BOT_DOWN_EXIT_CYCLES:
                    _note("bot down too long — monitor exiting")
                    return 0
            now = datetime.now()
            minute = now.hour * 60 + now.minute
            slots = sorted(
                set([h * 60 + m for h in range(24) for m in (0, 30)] + [23 * 60 + 50])
            )
            nxt = next((s for s in slots if s > minute), slots[0])
            delay = max(60, (nxt - minute) * 60)
            _note(f"check exit={code}; next in {delay // 60} min")
            time.sleep(delay)
    return run_once(date.today().isoformat(), _read(STATE_FILE))[0]


if __name__ == "__main__":
    sys.exit(main())
