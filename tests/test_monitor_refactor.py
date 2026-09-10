"""production_monitor decision contract after Inspection refactor."""
import importlib.util
from pathlib import Path

from hanoon_prime.inspection.checks import FAIL, OK, WARN, CheckResult
from hanoon_prime.inspection.joints import Manifest


def _load_monitor():
    p = Path(__file__).resolve().parents[1] / "scripts" / "production_monitor.py"
    spec = importlib.util.spec_from_file_location("monitor_refactor_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mf(*results: CheckResult) -> Manifest:
    return Manifest(ts=1.0, git_head="h", pid=1, results=tuple(results))


def test_decide_pass() -> None:
    m = _load_monitor()
    assert m._decide(_mf(CheckResult("telemetry", "health_ok", OK))) == 0


def test_decide_halim_dominates() -> None:
    m = _load_monitor()
    results = [
        CheckResult("halim", "halim_state_matches_clock", FAIL, detail="down"),
        CheckResult("telemetry", "health_ok", FAIL, detail="unreachable"),
    ]
    assert m._decide(_mf(*results)) == 3


def test_decide_hard_fail() -> None:
    m = _load_monitor()
    results = [
        CheckResult("telemetry", "health_ok", FAIL, detail="unreachable"),
        CheckResult("halim", "halim_state_matches_clock", OK, detail="state=ok"),
    ]
    assert m._decide(_mf(*results)) == 2


def test_decide_anomaly_only() -> None:
    m = _load_monitor()
    results = [
        CheckResult("processes", "bot_alive", FAIL, detail="not running"),
        CheckResult("halim", "halim_state_matches_clock", OK, detail="state=asleep"),
        CheckResult("identity", "bot_from_trusted_checkout", OK),
    ]
    assert m._decide(_mf(*results)) == 4


def test_decide_warn_is_pass() -> None:
    m = _load_monitor()
    results = [
        CheckResult(
            "execution_oracle", "enters_minted", WARN, detail="1 ENTER without fill"
        ),
        CheckResult("halim", "halim_state_matches_clock", OK, detail="state=asleep"),
        CheckResult("identity", "bot_from_trusted_checkout", OK),
    ]
    assert m._decide(_mf(*results)) == 0  # WARN-grade anomalies do not page
