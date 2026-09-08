# Brain-First Decision Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `juli.tick()` the single decision authority (IB → JULI → execution → learning) by absorbing every top-level gate into `brain/`, so a veto is always observable data in the INFO log, never a silent skip.

**Architecture:** The fast cortex (`NeuromorphicBrain`) turns *every evaluated ticker* into a `Verdict` (`ENTER`/`HOLD`/`VETOED` + reason + stage), applying policy from a new `brain/policy/` package: governor (pacing), trading_policy (direction/session/penny), portfolio_risk (slow-side manager + fast-side pure reader), safety (halt + pause + probe override). The slow cortex (`ConsolidationEngine`) owns all mutating policy state and publishes a `policy_state` dict into `BrainState`. `ib_cycle` becomes pure orchestration: assemble context, publish account facts, execute ENTER verdicts, reflect, journal, flatten (user commands).

**Tech Stack:** Python 3.11+, dataclasses, thread-safe `BrainState` dict exchange, existing `immune` constants, `SizingResult` from `brain/risk.py`, pytest (entry gates + regression), mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-09-brain-first-decision-pipeline-design.md` (approved design this plan implements — the plan argues from the spec, so both travel together; executors read both).

## Global Constraints

- **R1:** Only `cortex.py` produces `BUY`/`SELL`/`HOLD` verdicts; `cerebellum.py`, `edge.py`, `hands.py`, `hippocampus.py`, `immune.py` must never contain those strings. `VETOED` is a new brain-owned verdict string safe to use outside those modules.
- **R3 (per file, applies to all NEW modules and `juli.py` — none are in the skip list):** ≤200 lines/file, every function ≤40 lines, nesting ≤3 levels inside a function body.
- **R13 (applies to ALL source files incl. `ib_cycle.py`):** never compare a string literal against `BUY|SELL|HOLD|ENTER|EXIT|LONG|SHORT`. Compare against the imported action **constant** (`v.action == ENTER` — a `Name` comparator, not a `Constant`, is R13-safe). In *tests* string literals are fine.
- **R6/R14:** safety constants stay typed literals in `immune.py`; no `os.environ.getenv` near them.
- **R11/R15/R16/R18:** public functions docstred; no bare `except`; no `TODO`/`FIXME`; every module has a module docstring.
- **mypy strict** covers `src/hanoon_prime` (excludes `ib_cycle.py`, `ib_adapter.py`, etc.). All new `brain/policy/*` files and `juli.py` / `orchestrator.py` are type-checked.
- **Pre-commit black** runs on everything **EXCEPT `ib_cycle.py`** — never run black on `ib_cycle.py`.
- **pytest** gate: `pytest` runs the full suite; `tests/test_contract.py` must pass unedited (R3 200-line check will flag any new file over 200).
- Do **not** modify the learning pipeline, scanner behavior, sleep_manager (stays a top-level clock/data provider), or `tests/test_contract.py`.
- Verification baseline command: `pytest && mypy src/hanoon_prime && pre-commit run --all-files` (these must be green at `ace4da7` before Task 1).

## File Structure

New `src/hanoon_prime/brain/policy/` package:
- `verdict.py` — `Verdict` dataclass + `ENTER`/`HOLD`/`VETOED` constants + `to_dict()`.
- `governor.py` — `Governor`: per-cycle entry budget + per-ticker reuse cooldown (fast path).
- `trading_policy.py` — `TradingConfig` (moved from `config.py`, which becomes a re-export shim) + `TradingPolicy` helpers for penny bar / session / direction.
- `portfolio_risk.py` — `PortfolioRiskManager` **moved verbatim** from `monitor/portfolio_risk.py` (slow-side owner; mutable only on the slow cortex).
- `portfolio_gate.py` — pure **read-only** fast-side portfolio gate + size scaling, fed only by the published `policy_state` snapshot (same rules/order/reason strings as `pre_trade_risk_gate`/`adjust_size`).
- `safety.py` — `SafetyProducer` (slow-side): halt flag, pause, daily-loss / consecutive-loss / max-position nets, probe override hook, journal + telegram notification.

Modified:
- `brain/shared_state.py` — add default `policy_state`, `account_feed`, `policy_exits`, `consecutive_losses` keys.
- `brain/orchestrator.py` — `NeuromorphicBrain` gains `governor`, `decide_entry()`, `begin_entry_cycle()`, `note_entry()`, `resume()`, `set_safety_enabled()`; moves `_eval_one`'s scoring body here.
- `brain/consolidation.py` — slow cortex gains `_update_policy()` (portfolio + safety + publish) and owns `portfolio_risk` / `safety` instances.
- `juli.py` — `tick()`/`_evaluate_entries()` produce `Verdict`s; remove `_build_decision`/`_eval_one`; keep rotation; store recent verdicts; accept `session`.
- `ib_cycle.py` — remove all gates; add `_execute_verdict`, `_publish_account_feed`; `_finish_cycle` executes ENTER verdicts and journals them.
- `telemetry.py` — re-point `/health`, `/safety-net`, `/risk` to `policy_state`; POST toggles via brain commands; add `/verdicts`.
- `ib_adapter.py` — attach `journal` to the safety producer.
- `config.py` — becomes a re-export shim for `TradingConfig`/`TRADING_CONFIG`.
- `scripts/smoke_live.py` — decisions → `Verdict` assertions.
- `docs/ARCHITECTURE.md` — pipeline section rewrite.
- Delete `monitor/portfolio_risk.py` after Task 4.

Tests: new `tests/test_verdict.py`, `tests/test_governor.py`, `tests/test_trading_policy.py`, `tests/test_portfolio_risk.py`, `tests/test_portfolio_gate.py`, `tests/test_safety_producer.py`, `tests/test_brain_pipeline.py` (silent-block regression + decision gates), updates to `tests/test_telemetry.py`, `tests/test_bugfixes.py`, `tests/test_flow_throttle.py`.

### Task 1: `Verdict` dataclass

**Files:**
- Create: `src/hanoon_prime/brain/policy/__init__.py`
- Create: `src/hanoon_prime/brain/policy/verdict.py`
- Test: `tests/test_verdict.py`

**Interfaces:**
- Produces: `Verdict` (dataclass), constants `ENTER`, `HOLD`, `VETOED` (str), `Verdict.to_dict() -> dict[str, Any]`, `Verdict.as_execution_context() -> SimpleNamespace`. Every later task imports from `hanoon_prime.brain.policy.verdict`.
- Consumes: `SizingResult` from `hanoon_prime.brain.risk`.

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace
from hanoon_prime.brain.policy.verdict import ENTER, HOLD, VETOED, Verdict


def test_constants_are_distinct_verdict_strings():
    assert {ENTER, HOLD, VETOED} == {"ENTER", "HOLD", "VETOED"}


def test_verdict_to_dict_and_context():
    v = Verdict(
        ticker="NVD",
        action=ENTER,
        reason="ok",
        stage="governor",
        score=0.9,
        direction=1,
        horizon="scalp",
        thought=SimpleNamespace(direction=1, score=0.9),
    )
    d = v.to_dict()
    assert d["ticker"] == "NVD" and d["action"] == ENTER
    assert d["reason"] == "ok" and d["stage"] == "governor"
    ctx = v.as_execution_context()
    assert ctx.direction == 1 and ctx.score == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verdict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hanoon_prime.brain.policy'`

- [ ] **Step 3: Write implementation**

`src/hanoon_prime/brain/policy/__init__.py`:
```python
"""brain.policy — brain-owned decision policy package."""
```

`src/hanoon_prime/brain/policy/verdict.py`:
```python
"""brain.policy.verdict — the decision contract between brain and execution.

One Verdict is produced for EVERY evaluated ticker — never a silent
omission. A veto is data: the log line carries the stage that vetoed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..risk import SizingResult

ENTER: str = "ENTER"
HOLD: str = "HOLD"
VETOED: str = "VETOED"


@dataclass
class Verdict:
    """One brain decision for one ticker in one cycle."""

    ticker: str
    action: str = HOLD
    reason: str = ""
    stage: str = ""
    sizing: SizingResult | None = None
    stop: float | None = None
    target: float | None = None
    horizon: str = "scalp"
    score: float = 0.0
    direction: int = 0
    thought: Any = field(repr=False, default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for journal/telemetry (execution context excluded)."""
        return {
            "ticker": self.ticker,
            "action": self.action,
            "reason": self.reason,
            "stage": self.stage,
            "horizon": self.horizon,
            "score": round(float(self.score), 4),
            "direction": int(self.direction),
        }

    def as_execution_context(self) -> SimpleNamespace:
        """Rebuild the thought-shaped context place_bracket consumes."""
        return SimpleNamespace(
            direction=int(self.direction),
            score=float(self.score),
            verdict=self.action,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verdict.py -v && mypy src/hanoon_prime/brain/policy`
Expected: PASS (twotests), mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/brain/policy tests/test_verdict.py
git commit -m "feat(policy): brain policy verdict contract"
```

### Task 2: Pacing `Governor`

**Files:**
- Create: `src/hanoon_prime/brain/policy/governor.py`
- Test: `tests/test_governor.py`

**Interfaces:**
- Produces: `Governor` with `begin_cycle() -> None`, `may_enter(ticker: str) -> tuple[bool, str]`, `note_entry(ticker: str) -> None`. Consumes `MAX_ENTRIES_PER_CYCLE`, `ENTRY_REUSE_COOLDOWN_SEC` from `hanoon_prime.immune`.
- Ports `tests/test_flow_throttle.py` FIX-2026-09-08-02 throttle semantics (cap per cycle + per-ticker cooldown).

- [ ] **Step 1: Write the failing test**

```python
import time

from hanoon_prime.brain.policy.governor import Governor
from hanoon_prime.immune import ENTRY_REUSE_COOLDOWN_SEC, MAX_ENTRIES_PER_CYCLE


def test_budget_cap_rejects_after_limit():
    g = Governor()
    g.begin_cycle()
    for i in range(MAX_ENTRIES_PER_CYCLE):
        ok, _ = g.may_enter(f"T{i}")
        assert ok, f"T{i} should be admitted"
    ok, reason = g.may_enter("T_EXTRA")
    assert ok is False and reason == "cycle_budget"


def test_begin_cycle_resets_budget():
    g = Governor()
    g.begin_cycle()
    for i in range(MAX_ENTRIES_PER_CYCLE + 1):
        g.may_enter(f"T{i}")
    g.begin_cycle()
    ok, _ = g.may_enter("T_AGAIN")
    assert ok is True


def test_reuse_cooldown_blocks_reentry():
    g = Governor()
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True
    g.note_entry("NVD")
    g.begin_cycle()
    ok, reason = g.may_enter("NVD")
    assert ok is False and reason == "reuse_cooldown"


def test_cooldown_expires():
    g = Governor()
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True
    g.note_entry("NVD")
    g._last_entry["NVD"] = time.time() - (ENTRY_REUSE_COOLDOWN_SEC + 1.0)
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_governor.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write implementation**

```python
"""brain.policy.governor — entry pacing, not thinking.

Separate pacing from scoring: the governor caps how many entries may be
decided per cycle and how often a ticker may be re-entered. Single-threaded,
fast-path only, holds no locks.
"""

from __future__ import annotations

import time

from ..immune import ENTRY_REUSE_COOLDOWN_SEC, MAX_ENTRIES_PER_CYCLE


class Governor:
    """Per-cycle entry budget + per-ticker reuse cooldown."""

    def __init__(self) -> None:
        """Zero budget; cooldown timestamps empty."""
        self._cycle_used: int = 0
        self._last_entry: dict[str, float] = {}

    def begin_cycle(self) -> None:
        """Reset the per-cycle admitted count at the start of juli.tick."""
        self._cycle_used = 0

    def may_enter(self, ticker: str) -> tuple[bool, str]:
        """(True, "ok") when the ticker may be decided as an entry this cycle.

        Consumes the cycle budget on approval; the reuse cooldown is keyed
        by ticker and only written when a bracket is actually placed.
        """
        if self._cycle_used >= MAX_ENTRIES_PER_CYCLE:
            return False, "cycle_budget"
        if time.time() - self._last_entry.get(ticker, 0.0) < ENTRY_REUSE_COOLDOWN_SEC:
            return False, "reuse_cooldown"
        self._cycle_used += 1
        return True, "ok"

    def note_entry(self, ticker: str) -> None:
        """Stamp the cooldown AFTER a real bracket was placed."""
        self._last_entry[ticker] = time.time()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_governor.py -v && mypy src/hanoon_prime/brain/policy/governor.py`
Expected: PASS, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/brain/policy/governor.py tests/test_governor.py
git commit -m "feat(policy): entry governor (cycle budget + cooldown)"
```

### Task 3: `TradingPolicy` (config home moves into `brain/`)

**Files:**
- Create: `src/hanoon_prime/brain/policy/trading_policy.py`
- Modify: `src/hanoon_prime/config.py` (shim re-export)
- Test: `tests/test_trading_policy.py`

**Interfaces:**
- Produces: `TradingConfig` dataclass (fields/methods **identical** to today's `config.TradingConfig`), singleton `TRADING_CONFIG` — now importable from BOTH `hanoon_prime.config` (shim) and `hanoon_prime.brain.policy.trading_policy`. New pure method `TradingConfig.is_penny_bar_cleared(ticker: str, price: float, score: float) -> tuple[bool, str]` returns `(False, "low_penny_score")` when `price < PENNY_PRICE and abs(score) < PENNY_SCORE_BAR`.
- Consumes: `PENNY_PRICE`, `PENNY_SCORE_BAR` from `hanoon_prime.immune`.
- Rationale: telemetry `POST/GET /config` and `ib_cycle._check_eod_flatten` keep using the SAME singleton object; only its definition home moves into `brain/`. All existing `from .config import TRADING_CONFIG` imports keep working via the shim.

- [ ] **Step 1: Write the failing test**

```python
from hanoon_prime.brain.policy.trading_policy import TRADING_CONFIG


def test_direction_helpers_unchanged():
    assert TRADING_CONFIG.direction_mode == "long_only"
    assert TRADING_CONFIG.is_direction_allowed("BUY")
    assert not TRADING_CONFIG.is_direction_allowed("SELL")


def test_session_helper_matches_today():
    assert TRADING_CONFIG.is_session_active("rth") is True


def test_penny_bar_raises_bar_for_low_score_pennies():
    ok, reason = TRADING_CONFIG.is_penny_bar_cleared("PENN", 0.80, 0.5)
    assert ok is False and reason == "low_penny_score"
    assert TRADING_CONFIG.is_penny_bar_cleared("PENN", 0.80, 0.9)[0] is True
    assert TRADING_CONFIG.is_penny_bar_cleared("AAPL", 150.0, 0.5)[0] is True


def test_config_shim_re_exports_singleton():
    from hanoon_prime.config import TRADING_CONFIG as SHIM

    assert SHIM is TRADING_CONFIG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_policy.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write implementation**

`src/hanoon_prime/brain/policy/trading_policy.py` — move the dataclass verbatim from `config.py`, add `is_penny_bar_cleared`, and import `PENNY_PRICE`, `PENNY_SCORE_BAR`:

```python
"""brain.policy.trading_policy — decision parts of the trading configuration.

Direction mode, session enablement, and the sub-dollar confidence bar.
The singleton is the SAME object `config.TRADING_CONFIG` re-exports, so
telemetry and existing code share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...immune import PENNY_PRICE, PENNY_SCORE_BAR


@dataclass
class TradingConfig:
    """Decision-relevant trading configuration (brain-owned)."""

    session_pre_market: bool = True
    session_rth: bool = True
    session_post_market: bool = True
    session_overnight: bool = True

    direction_mode: str = "long_only"

    eod_flatten_enabled: bool = True
    eod_flatten_minutes: float = 5.0

    horizons: set[str] = field(default_factory=lambda: {"scalp"})

    def is_session_active(self, session: str) -> bool:
        """Check if a session is enabled."""
        return getattr(self, f"session_{session}", True)

    def is_direction_allowed(self, side: str) -> bool:
        """Check if a trade side is allowed."""
        if self.direction_mode == "both":
            return True
        if self.direction_mode == "long_only":
            return side.upper() in ("BUY", "LONG")
        if self.direction_mode == "short_only":
            return side.upper() in ("SELL", "SHORT")
        return True

    def is_penny_bar_cleared(self, ticker: str, price: float, score: float) -> tuple[bool, str]:
        """Raise-the-bar for sub-dollar tickers: return (False, reason) if
        the brain must be EXTREMELY sure of a micro-cap scalp (not a hard
        price block — a higher bar the score can still clear)."""
        if price < PENNY_PRICE and abs(score) < PENNY_SCORE_BAR:
            return False, "low_penny_score"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for telemetry."""
        return {
            "sessions": {
                "pre_market": self.session_pre_market,
                "rth": self.session_rth,
                "post_market": self.session_post_market,
                "overnight": self.session_overnight,
            },
            "direction_mode": self.direction_mode,
            "eod_flatten_enabled": self.eod_flatten_enabled,
            "eod_flatten_minutes": self.eod_flatten_minutes,
            "horizons": sorted(self.horizons),
        }


TRADING_CONFIG = TradingConfig()

__all__ = ["TradingConfig", "TRADING_CONFIG"]
```

`src/hanoon_prime/config.py` — replace the dataclass body with a re-export shim:

```python
"""config.py — re-export shim for the brain-owned trading policy.

The decision configuration now lives in ``brain/policy/trading_policy.py``;
this module keeps the historical import path working.
"""

from __future__ import annotations

from .brain.policy.trading_policy import TradingConfig, TRADING_CONFIG

__all__ = ["TradingConfig", "TRADING_CONFIG"]
```

- [ ] **Step 4: Run test to verify it passes, then check R3 + import blast radius**

Run: `pytest tests/test_trading_policy.py tests/test_flow_throttle.py::TestLongOnlyDefault tests/test_bugfixes.py::TestBug3OffMarketGuard -v && python -c "from hanoon_prime.config import TRADING_CONFIG; print(TRADING_CONFIG.direction_mode)"`
Expected: PASS; prints `long_only` (no circular-import crash).

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/brain/policy/trading_policy.py src/hanoon_prime/config.py tests/test_trading_policy.py
git commit -m "feat(policy): trading config moves into brain/policy (shim keeps imports)"
```

### Task 4: Move `PortfolioRiskManager` into `brain/policy/`

**Files:**
- Create: `src/hanoon_prime/brain/policy/portfolio_risk.py` (copy of the CURRENT file)
- Delete: `src/hanoon_prime/monitor/portfolio_risk.py`
- Test: `tests/test_portfolio_risk.py`

**Interfaces:**
- Produces: `PortfolioRiskManager` (API unchanged: `update_equity`, `update_positions`, `total_exposure`, `pre_trade_risk_gate`, `adjust_size`, `check_portfolio_giveback`, `get_risk_state`), plus `PortfolioRiskState`, `GivebackDecision`, constants. Importers switch to `hanoon_prime.brain.policy.portfolio_risk`.
- Consumes: `MAX_CONCURRENT_POSITIONS`, `MAX_POSITION_NOTIONAL` from `hanoon_prime.immune`, via `from ...immune import`.
- Behavior-preserving: file content is byte-identical except the two relative import dot-counts and the module docstring's first line path.

- [ ] **Step 1: Move the file and fix imports**

Create `src/hanoon_prime/brain/policy/portfolio_risk.py` with the **exact current contents** of `monitor/portfolio_risk.py`, changing:
- docstring line 1 → `"""brain.policy.portfolio_risk — portfolio-level risk (rebuild risk/portfolio.py port).`
- line 14 import → `from ...immune import MAX_CONCURRENT_POSITIONS, MAX_POSITION_NOTIONAL`

Then delete `monitor/portfolio_risk.py`:
```bash
git mv src/hanoon_prime/monitor/portfolio_risk.py src/hanoon_prime/brain/policy/portfolio_risk.py
```

- [ ] **Step 2: Write the failing port test**

```python
from hanoon_prime.brain.policy.portfolio_risk import (
    CONCENTRATION_CAP,
    PortfolioRiskManager,
)


def _hol(pnl=0.0, value=0.0):
    return {"pnl": pnl, "value": value, "pct": 0.0}


def test_equity_unsynced_blocks_entries():
    m = PortfolioRiskManager()
    assert m.pre_trade_risk_gate("TSLA", 1000.0)[0] is False


def test_scalar_drawdown_rule():
    m = PortfolioRiskManager()
    m.update_equity(100_000.0)
    m.update_equity(70_000.0)  # 30% drawdown -> stress + scalar floor
    assert m.get_risk_state()["drawdown"] > 0.20
    assert m.get_risk_state()["risk_scalar"] < 0.5
    allowed, reason = m.pre_trade_risk_gate("TSLA", 1_000.0)
    assert allowed is False and "stress" in reason


def test_concentration_cap():
    m = PortfolioRiskManager()
    m.update_equity(100_000.0)
    allowed, reason = m.pre_trade_risk_gate("NVD", 100_000.0 * CONCENTRATION_CAP * 1.01)
    assert allowed is False and reason.startswith("concentration")


def test_adjust_size_scales_by_risk_scalar():
    m = PortfolioRiskManager()
    m.update_equity(100_000.0)
    assert m.adjust_size(100, 10.0) == 100  # scalar 1.0
    m.update_equity(85_000.0)
    assert m.adjust_size(100, 10.0) < 100  # scalar < 1.0
```

- [ ] **Step 3: Run to verify import + behavior**

Run: `pytest tests/test_portfolio_risk.py -v`
Expected: PASS. Then grep for stale importers:
Run: `rg -l "monitor.portfolio_risk|monitor import portfolio_risk|from .monitor.portfolio_risk" src tests`
Expected: only `brain/policy/portfolio_risk.py` internal references remain (if `telemetry.py`/`ib_cycle.py` still import it, note them for Tasks 8–9; do NOT fix them yet — they still resolve).

- [ ] **Step 4: R3 + mypy gate**

Run: `python -c "import pathlib; print(len(pathlib.Path('src/hanoon_prime/brain/policy/portfolio_risk.py').read_text().splitlines()))"` and `mypy src/hanoon_prime/brain/policy/portfolio_risk.py`
Expected: `200` (≤200) and mypy clean.

- [ ] **Step 5: Commit**

```bash
git add -A src/hanoon_prime/brain/policy/portfolio_risk.py tests/test_portfolio_risk.py
git commit -m "refactor(policy): portfolio risk moves into brain/policy (behavior-preserving)"
```

### Task 5: Pure fast-side portfolio gate

**Files:**
- Create: `src/hanoon_prime/brain/policy/portfolio_gate.py`
- Test: `tests/test_portfolio_gate.py`

**Interfaces:**
- Produces: `portfolio_gate(ticker: str, notional: float, portfolio: dict[str, Any]) -> tuple[bool, str]` and `scale_shares(shares: int, price: float, risk_scalar: float, exposure: float) -> int`. Both are PURE — they only read the published `portfolio` dict from `policy_state`; the fast cortex never mutates slow-side state.
- Consumes: `CONCENTRATION_CAP`, `STRESS_SIZE_CAP`, `STRESS_DRAWDOWN`, `RISK_MIN`... borrowed as constants in the module, and the published fields `equity`, `equity_synced`, `risk_scalar`, `drawdown`, `stress_mode`, `exposure`, `position_count`, `max_positions`, `holdings` (dict `{sym: value}`).
- Mirrors `pre_trade_risk_gate`/`adjust_size` exactly (same checks, same reason strings, same order).

- [ ] **Step 1: Write the failing test**

```python
from hanoon_prime.brain.policy.portfolio_gate import portfolio_gate, scale_shares

BASE = {
    "equity": 100_000.0,
    "equity_synced": True,
    "risk_scalar": 1.0,
    "drawdown": 0.0,
    "stress_mode": False,
    "exposure": 0.0,
    "position_count": 0,
    "max_positions": 5,
    "holdings": {},
}


def test_equity_unsynced_rejects():
    p = dict(BASE, equity_synced=False)
    ok, reason = portfolio_gate("TSLA", 1000.0, p)
    assert ok is False and reason == "equity_unsynced"


def test_concentration_rejects():
    p = dict(BASE, holdings={"NVD": 0.0})
    ok, reason = portfolio_gate("NVD", 30_000.0, p)
    assert ok is False and reason.startswith("concentration")


def test_max_positions_rejects():
    p = dict(BASE, position_count=5, max_positions=5)
    assert portfolio_gate("TSLA", 1000.0, p)[0] is False


def test_exposure_cap_rejects():
    p = dict(BASE, exposure=1.0)
    assert portfolio_gate("TSLA", 1000.0, p)[0] is False


def test_scale_shares_dampens():
    assert scale_shares(100, 10.0, 1.0, 0.0) == 100
    assert scale_shares(100, 10.0, 0.5, 0.0) == 50
    assert scale_shares(100, 10.0, 1.0, 0.9) < 100


def test_no_portfolio_field_is_never_authoritative():
    p = dict(BASE, equity_synced=False, holdings={})
    assert portfolio_gate("TSLA", 1000.0, p)[0] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_portfolio_gate.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write implementation**

```python
"""brain.policy.portfolio_gate — fast-side READ-ONLY portfolio checks.

The slow cortex owns the mutable PortfolioRiskManager; its published
`policy_state.portfolio` snapshot is the ONLY thing the fast cortex reads.
These pure functions faithfully reproduce pre_trade_risk_gate / adjust_size
checks (same reasons, same order) so the fast path never mutates state.
"""

from __future__ import annotations

from typing import Any

from ...immune import MAX_POSITION_NOTIONAL, MAX_CONCURRENT_POSITIONS

CONCENTRATION_CAP: float = 0.25
BUDGET: float = MAX_POSITION_NOTIONAL * MAX_CONCURRENT_POSITIONS


def portfolio_gate(ticker: str, notional: float, portfolio: dict[str, Any]) -> tuple[bool, str]:
    """(True, "") to admit, (False, reason) to veto, from the snapshot."""
    if not portfolio.get("equity_synced", False) or float(portfolio.get("equity", 0.0)) <= 0:
        return False, "equity_unsynced"
    if portfolio.get("stress_mode", False) and notional > BUDGET * 0.30:
        return False, "stress_size>30%"
    if float(portfolio.get("exposure", 0.0)) >= 1.0:
        return False, "exposure_cap"
    holdings = portfolio.get("holdings", {}) or {}
    held = abs(float(holdings.get(ticker, 0.0) or 0.0))
    conc = (held + notional) / float(portfolio["equity"])
    if conc > CONCENTRATION_CAP:
        return False, f"concentration={conc:.2f}"
    if int(portfolio.get("position_count", 0)) >= int(portfolio.get("max_positions", MAX_CONCURRENT_POSITIONS)):
        return False, f"max_positions={int(portfolio.get('position_count', 0))}"
    if portfolio.get("stress_mode", False):
        return False, f"stress_mode dd={float(portfolio.get('drawdown', 0.0)):.2f}"
    if float(portfolio.get("risk_scalar", 1.0)) <= 0.15:
        return False, f"risk_scalar={float(portfolio.get('risk_scalar', 1.0)):.2f}"
    return True, ""


def scale_shares(shares: int, price: float, risk_scalar: float, exposure: float) -> int:
    """Mirror adjust_size: risk scalar + exposure dampening."""
    if shares <= 0 or price <= 0:
        return shares
    damp = max(0.5, 1.0 - (abs(float(exposure)) / max(BUDGET, 1.0)) * 0.5)
    return int(shares * price * float(risk_scalar) * damp / price)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_portfolio_gate.py -v && mypy src/hanoon_prime/brain/policy/portfolio_gate.py`
Expected: PASS, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/brain/policy/portfolio_gate.py tests/test_portfolio_gate.py
git commit -m "feat(policy): pure fast-side portfolio gate from published snapshot"
```

### Task 6: `SafetyProducer` (halt + pause + probe override)

**Files:**
- Create: `src/hanoon_prime/brain/policy/safety.py`
- Test: `tests/test_safety_producer.py`

**Interfaces:**
- Produces: `SafetyProducer` with lifecycle run on the SLOW cortex: `begin_call() -> None`, `on_daily_pnl(pnl: float) -> None`, `on_consecutive_losses(n: int) -> None`, `on_position_count(n: int) -> None`, `authorized() -> tuple[bool, str]`, `resume() -> None`, `set_enabled(en: bool) -> None`, `attach_journal(journal) -> None`, plus read-only props `halted: bool`, `enabled: bool`. Every halt calls the existing `_telegram.safety_halt` and journals `{"event": "halt", "reason": ...}`.
- Consumes: `DAILY_LOSS_LIMIT`, `CONSECUTIVE_LOSSES_PAUSE`, `MAX_CONCURRENT_POSITIONS` from `hanoon_prime.immune`, `safety_halt` from `hanoon_prime._telegram`.
- Consistency note: the pause behavior (self-pause `PAUSE_DURATION_MIN` bars) currently lives in `Hippocampus.check_entry_allowed` for the sim path (`hands.py`); it STAYS there untouched. The live halt policy moves here.

- [ ] **Step 1: Write the failing test**

```python
from hanoon_prime.brain.policy.safety import SafetyProducer
from hanoon_prime.immune import CONSECUTIVE_LOSSES_PAUSE, DAILY_LOSS_LIMIT, MAX_CONCURRENT_POSITIONS


def _producer():
    s = SafetyProducer()
    s.begin_call()
    return s


def test_defaults_authorized():
    s = _producer()
    ok, reason = s.authorized()
    assert ok is True and reason == ""


def test_daily_loss_trips_halt():
    s = _producer()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    ok, reason = s.authorized()
    assert ok is False and "daily" in reason
    assert s.halted is True


def test_consecutive_losses_trips_halt():
    s = _producer()
    s.on_consecutive_losses(CONSECUTIVE_LOSSES_PAUSE)
    assert s.authorized()[0] is False


def test_position_count_trips_halt():
    s = _producer()
    s.on_position_count(MAX_CONCURRENT_POSITIONS + 1)
    assert s.authorized()[0] is False


def test_resume_clears_halt():
    s = _producer()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    s.resume()
    assert s.authorized()[0] is True and s.halted is False


def test_disabled_never_halts():
    s = SafetyProducer()
    s.set_enabled(False)
    s.begin_call()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT * 3))
    assert s.authorized()[0] is True


def test_halt_journals_and_notifies():
    events = []
    s = SafetyProducer(journal=None)
    s_append = events.append
    s._journal = type("J", (), {"append": lambda self, e: s_append(e)})()
    s._notify = lambda reason: s_append(("notify", reason))
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    assert any(e.get("event") == "halt" for e in events)
    assert any(k == "notify" for k, _ in events if isinstance(k, str))


def test_pause_reason_survives_reauthorize():
    s = _producer()
    s.on_position_count(MAX_CONCURRENT_POSITIONS + 1)
    _, reason = s.authorized()
    assert reason != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_safety_producer.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write implementation**

```python
"""brain.policy.safety — live halt + pause policy (slow-cortex owned).

Safety nets are portfolio-level and low-cadence, so the slow cortex feeds
them from the account feed and publishes `authorized` in `policy_state`.
Halt behavior is preserved from ib_cycle._halt: block NEW entries, never
stop the bot, notify + journal.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ..._telegram import safety_halt as _telegram_halt
from ...immune import (
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    MAX_CONCURRENT_POSITIONS,
)

log = logging.getLogger(__name__)


class SafetyProducer:
    """Feed-run halt/pause policy; publishes authorization each cycle."""

    def __init__(self, journal: Any = None, notify: Callable[[str], None] | None = None) -> None:
        """Start enabled, unhalted, no pause reason."""
        self.enabled: bool = True
        self.halted: bool = False
        self.pause_reason: str = ""
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._position_count: int = 0
        self._journal = journal
        self._notify = notify or _telegram_halt

    def attach_journal(self, journal: Any) -> None:
        """Wire the bot journal for halt events."""
        self._journal = journal

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the safety system (webapp command)."""
        self.enabled = enabled
        if enabled is False:
            self.resume()

    def begin_call(self) -> None:
        """Start one slow-cortex policy pulse."""
        self.pause_reason = ""

    def on_daily_pnl(self, pnl: float) -> None:
        """Feed IB day P&L."""
        self._daily_pnl = float(pnl)

    def on_consecutive_losses(self, n: int) -> None:
        """Feed current loss streak."""
        self._consecutive_losses = int(n)

    def on_position_count(self, n: int) -> None:
        """Feed current open position count."""
        self._position_count = int(n)

    def authorized(self) -> tuple[bool, str]:
        """(True, "") to allow entries; (False, reason) to veto."""
        if not self.enabled:
            return True, ""
        if self.halted:
            return False, self.pause_reason
        if self._daily_pnl < -DAILY_LOSS_LIMIT:
            return self._halt("daily_loss_limit")
        if self._consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE:
            return self._halt("consecutive_losses")
        if self._position_count > MAX_CONCURRENT_POSITIONS:
            return self._halt("too_many_positions")
        return True, ""

    def resume(self) -> None:
        """Clear a halt (webapp resume command)."""
        if self.halted:
            log.info("SAFETY: halt cleared")
        self.halted = False
        self.pause_reason = ""

    def _halt(self, reason: str) -> tuple[bool, str]:
        """Trip the halt once: flag, journal, notify."""
        self.halted = True
        self.pause_reason = reason
        log.critical("SAFETY HALT: %s", reason)
        if self._journal is not None:
            try:
                self._journal.append({"event": "halt", "reason": reason, "ts": time.time()})
            except Exception as exc:
                log.warning("Safety journal failed: %s", exc)
        try:
            self._notify(reason)
        except Exception as exc:
            log.warning("Safety notify failed: %s", exc)
        return False, reason
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_safety_producer.py -v && mypy src/hanoon_prime/brain/policy/safety.py`
Expected: PASS, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/brain/policy/safety.py tests/test_safety_producer.py
git commit -m "feat(policy): safety producer owns halt, pause, and probe override state"
```

### Task 7: BrainState keys + slow-cortex `_update_policy`

**Files:**
- Modify: `src/hanoon_prime/brain/shared_state.py`
- Modify: `src/hanoon_prime/brain/consolidation.py`
- Test: `tests/test_brain_policy_cycle.py`

**Interfaces:**
- Produces: new `BrainState` default keys —
  - `policy_state = DEFAULT_POLICY_STATE` (dict): `authorized`, `enabled`, `halted`, `pause_reason`, `risk_scalar`, `equity_synced`, `drawdown`, `exposure`, `stress_mode`, `position_count`, `max_positions`, `daily_pnl`, `consecutive_losses`, `equity`, `holdings`.
  - `policy_exits = []` (list of `{"ticker", "reason", "type", "ts"}` objects)
  - `account_feed = {}`
  - `consecutive_losses = 0`
- `ConsolidationEngine` gains `self.portfolio_risk: PortfolioRiskManager` and `self.safety: SafetyProducer`, plus `_update_policy()`:
  1. `feed = dict(self.brain_state.get("account_feed", {}))` — no-op when empty.
  2. `self.safety.begin_call()`
  3. `self.safety.on_daily_pnl(feed.get("daily_pnl", 0.0))`
  4. `self.safety.on_consecutive_losses(self.brain_state.get("consecutive_losses", 0))`
  5. `self.safety.on_position_count(len(self.brain_state.get("positions_open", {})))`
  6. `self.portfolio_risk.update_equity(feed.get("equity"))` when `feed["equity"]` present (also sets `equity_synced=True`)
  7. `self.portfolio_risk.update_positions(feed["positions"])` when present
  8. givebacks = `self.portfolio_risk.check_portfolio_giveback()` → publish into `policy_exits` with `type="portfolio_giveback"`
  9. publish `policy_state` = `{**self.portfolio_risk.get_risk_state(), **safety_auth}` where `safety_auth = {"authorized", "enabled", "halted", "pause_reason"}` from `self.safety.authorized()`
  - Ordering: `safety.authorized()` is called LAST so a halt/watch reason from THIS pulse is in the published `policy_state` immediately (matches today's semantics: `_check_safety(pnl)` then `_sync_portfolio_risk()` each cycle).
- Consumes: `PortfolioRiskManager` + `SafetyProducer` from Tasks 4/6; existing `consolidation.py` interval (30s) and `_update_regime/_update_halim/_run_thinker` untouched.
- Plugs into the existing `_cycle`: add `self._update_policy()` as a new step (near the end); do NOT alter regime/thinker/news steps.

- [ ] **Step 1: Write the failing test**

```python
import time
from hanoon_prime.brain.consolidation import ConsolidationEngine
from hanoon_prime.brain.shared_state import BrainState, DEFAULT_POLICY_STATE
from hanoon_prime.immune import DAILY_LOSS_LIMIT


def _engine():
    state = BrainState()
    eng = ConsolidationEngine(state)
    eng._update_policy()
    return eng, state


def test_update_policy_from_account_feed_publishes():
    eng, state = _engine()
    state.update(
        account_feed={"daily_pnl": 100.0, "equity": 100_000.0, "positions": {}},
        positions_open={},
        consecutive_losses=0,
    )
    eng._update_policy()
    ps = state.get("policy_state")
    assert ps["equity_synced"] is True
    assert ps["authorized"] is True
    assert ps["daily_pnl"] == 100.0


def test_daily_loss_halt_shows_in_published_state():
    eng, state = _engine()
    state.update(account_feed={"daily_pnl": -(DAILY_LOSS_LIMIT + 1.0)}, consecutive_losses=0, positions_open={})
    eng._update_policy()
    ps = state.get("policy_state")
    assert ps["authorized"] is False
    assert ps["halted"] is True
    assert "daily" in ps["pause_reason"]


def test_giveback_published_as_policy_exit():
    eng, state = _engine()
    state.update(
        account_feed={"daily_pnl": 0.0, "equity": 100_000.0, "positions": {"NVD": {"pnl": 40_000.0, "value": 80_000.0, "pct": 0.0}}},
        consecutive_losses=0,
        positions_open={"NVD": {}},
    )
    eng._update_policy()
    exits = state.get("policy_exits")
    assert any(e.get("type") == "portfolio_giveback" for e in exits)


def test_default_policy_state_authorized_but_unsynced():
    assert DEFAULT_POLICY_STATE["authorized"] is True
    assert DEFAULT_POLICY_STATE["equity_synced"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brain_policy_cycle.py -v`
Expected: FAIL — import error (`DEFAULT_POLICY_STATE`, `_update_policy` missing).

- [ ] **Step 3: Implement shared_state additions**

In `src/hanoon_prime/brain/shared_state.py`, add a `DEFAULT_POLICY_STATE` module constant (plain dict) and register new keys in the `BrainState` snapshot/init alongside the existing ones:

```python
DEFAULT_POLICY_STATE = {
    "authorized": True,
    "enabled": True,
    "halted": False,
    "pause_reason": "",
    "risk_scalar": 1.0,
    "equity_synced": False,
    "drawdown": 0.0,
    "exposure": 0.0,
    "stress_mode": False,
    "position_count": 0,
    "max_positions": 0,
    "daily_pnl": 0.0,
    "consecutive_losses": 0,
    "equity": 0.0,
    "holdings": {},
}
```

Add to the `BrainState` initialization (alongside `_health`/`_risk_state`, etc.): `policy_state` (deep-copied dict), `policy_exits` (list), `account_feed` (dict), `consecutive_losses` (int) — all with thread-safe readers/writers matching the existing pattern (a `get()` that returns a fresh deep copy and an `update(**kwargs)`).

- [ ] **Step 4: Implement `_update_policy` in ConsolidationEngine**

Add at class top: `self.portfolio_risk = PortfolioRiskManager()` and `self.safety = SafetyProducer()` (both importable from the policy package — pass `journal=None`; ib_adapter wires the journal in Task 11). Add in `_cycle` after the existing steps: `self._update_policy()`. Implement:

```python
def _update_policy(self) -> None:
    """Push account facts through portfolio + safety policy."""
    feed = self.brain_state.get("account_feed")
    self.safety.begin_call()
    self.safety.on_daily_pnl(feed.get("daily_pnl", 0.0))
    self.safety.on_consecutive_losses(self.brain_state.get("consecutive_losses", 0))
    self.safety.on_position_count(len(self.brain_state.get("positions_open", {})))
    equity = feed.get("equity")
    positions = feed.get("positions")
    if equity is not None:
        self.portfolio_risk.update_equity(equity)
    if positions:
        self.portfolio_risk.update_positions(positions)
    givebacks = self.portfolio_risk.check_portfolio_giveback()
    risk = self.portfolio_risk.get_risk_state()
    auth, reason = self.safety.authorized()
    self.brain_state.update(
        policy_state={
            **risk,
            "authorized": auth,
            "enabled": self.safety.enabled,
            "halted": self.safety.halted,
            "pause_reason": reason,
            "consecutive_losses": self.safety._consecutive_losses,
            "daily_pnl": self.safety._daily_pnl,
        },
        policy_exits=[
            {"ticker": g.ticker, "reason": g.reason, "type": "portfolio_giveback", "ts": time.time()}
            for g in givebacks
        ],
    )
```

(`risk` from `get_risk_state()` already carries `risk_scalar/drawdown/exposure/stress_mode/equity/equity_synced/position_count/max_positions/holdings`.) Check `get_risk_state()` — if it lacks `holdings`/`posiotion_count`, extend it in Task 4's class (verify against `monitor/portfolio_risk.py`; keep ≤200 lines by trimming a docstring if needed).

- [ ] **Step 5: Run test to verify it passes + regression sweep**

Run: `pytest tests/test_brain_policy_cycle.py -v && pytest tests/test_contract.py tests/test_neuromorphic.py -q && mypy src/hanoon_prime/brain`
Expected: PASS; contract + neuromorphic suite still green (no `policy_state` key collisions).

- [ ] **Step 6: Commit**

```bash
git add src/hanoon_prime/brain/shared_state.py src/hanoon_prime/brain/consolidation.py tests/test_brain_policy_cycle.py
git commit -m "feat(brain): policy_state/policy_exits publish on slow cortex"
```

### Task 8: `NeuromorphicBrain.decide_entry` — the single decision point

**Files:**
- Modify: `src/hanoon_prime/brain/orchestrator.py`
- Test: `tests/test_brain_pipeline.py` (the silent-block regression + full gate order)

**Interfaces:**
- Produces (all on `NeuromorphicBrain`):
  - `__init__` builds `self.governor = Governor()`, `self.probe = ProbeRecovery()` (replaces the module singleton in the brain), `self.trading_policy = TRADING_CONFIG` (module-level for reader convenience — cached at construction).
  - `begin_entry_cycle() -> None` → `self.governor.begin_cycle()` (called by `juli.tick`; spec: `begin_entry_cycle()`).
  - `decide_entry(ticker, snap, open_positions, session="rth") -> Verdict` — THE single decision point (replaces `juli._build_decision`/`_eval_one`; spec: `NeuromorphicBrain.decide_entry`).
  - `note_entry(ticker) -> None` → `self.governor.note_entry(ticker)` (spec: `_execute_verdict feeds governor.note_entry`).
  - `resume() -> None` → clears brain-side pause state (telemetry POST) — if the old `self.paused`-style state exists, clear it; the authoritative halt lives in `policy_state` via SafetyProducer.
  - `set_safety_enabled(enabled: bool) -> None` — stored in `brain_state` so the slow cortex can honor it next pulse.
  - Keeps: `register_position`, `position_closed`, `_learn_from_real`, `note_eval_failure`, `on_intense_burst`, `tick()`, `BrainState` exchange, all existing neuromorphic subsystems. Does NOT touch the learning loop.
- Consumes: `compute_alpha_from_snap` + `entry_bars` from `hanoon_prime.juli_feed` (fast dead-idle gate), `check_tick_latency` from `hanoon_prime.cerebellum`, `SizingResult` from `.risk`, `Verdict`/`ENTER`/`HOLD`/`VETOED` from `.policy.verdict`, `portfolio_gate`/`scale_shares` from `.policy.portfolio_gate`, `TRADING_CONFIG` from `.policy.trading_policy`.
- **Gate order (every evaluated ticker returns a Verdict — semantics identical to today's silent gates, now observable):**
  1. snap validity: `snap is None` or `len(snap.get("prices", [])) < 20` or NaN on `last/bid/ask/mid` → `VETOED reason="no_data" stage="validity"`
  2. `set_latest_prices(ticker, snap["prices"])`
  3. alpha via `compute_alpha_from_snap(ticker, snap)`; `entry_bars` from `juli_feed.entry_bars`; `latency_ms = check_tick_latency(snap)` when present
  4. `result = self.tick(ticker, snap)` (fast cortex signal — same as today, returns `(action, signals, thought)`); **exceptions → `VETOED reason="eval_error" stage="pipeline"`** + `note_eval_failure`
  5. `direction == 0` → `HOLD reason="no_signal"`
  6. `is_session_active(session)` False → `VETOED reason="session_disabled" stage="trading_policy"`
  7. `is_direction_allowed(direction)` False → `VETOED reason="direction_rejected" stage="trading_policy"`
  8. penny bar: `trading_policy.is_penny_bar_cleared(...)` → `VETOED reason="low_penny_score" stage="trading_policy"`
  9. `policy_state = self.state.get("policy_state", DEFAULT_POLICY_STATE)`; if `authorized is False` → **probe override**: `self.probe.maybe_probe(score, bid, ask, policy_state.get("consecutive_losses", 0))` → if probe says enter: `ENTER reason="probe_recovery" stage="probe_recovery"` (keeps ProbeRecovery brain-owned and still able to override a HOLD-level halt); else `VETOED reason=(pause_reason or "halted") stage="safety"`
  10. `governor.may_enter(ticker)` → `VETOED reason∈{"cycle_budget","reuse_cooldown"} stage="governor"`
  11. build `SizingResult` via `self.sizing(ticker, snap, thought)`; `None or shares<=0` → `HOLD reason="not_sized"`
  12. pure portfolio gate: `portfolio_gate(ticker, notional, portfolio_snapshot)` → `VETOED reason stage="portfolio_risk"`; then `shares = scale_shares(...)`; if `shares<=0` → `VETOED reason="sized_to_zero" stage="portfolio_risk"`
  13. return `ENTER` Verdict carrying `sizing/stop/target/horizon/score/direction/thought`
  - A `Session US session` INFO line mirrors today's `THINK ≥... US session annealed...` (single-line logger).

Helper split (all ≤40 lines, in orchestrator.py — which is R3 200-line-skipped, so only function length matters):
```python
def _check_snapshot_valid(self, snap: dict[str, Any] | None) -> str:
    """Return a veto reason when the snapshot is unusable, else ''."""
def _apply_fast_gates(self, ticker, price, score, direction, session, portfolio) -> Verdict | None:
    """Apply policy gates that need no governor/sizing; None = admitted."""
def _build_enter(self, ticker, snap, thought, sizing, direction, price) -> Verdict:
    """Scale by risk and build the ENTER Verdict."""
```
(These may live in orchestrator.py — mocked elegantly — or as private methods; keep total added lines under 200. Prefer keeping the scoring body in `decide_entry` itself and gate helpers tiny.)

- [ ] **Step 1: Write the failing regression test** (silent-block prevention; `tests/test_brain_pipeline.py`)

```python
from types import SimpleNamespace
from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.policy.verdict import ENTER, HOLD, VETOED, Verdict
from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE


def _snap():
    import time, math
    return {
        "prices": [1.0] * 40,
        "bid": 100.0, "ask": 100.1, "mid": 100.05, "last": 100.0,
        "ts": time.time(), "volume": 1000,
    }


def _brain():
    b = NeuromorphicBrain(enable_neuromorphic=False)
    b.tick = lambda *a, **k: _fake_result()
    return b


def _fake_result(side="BUY", score=0.9, direction=1):
    return {
        "price": 100.0,
        "verdict": side,
        "signals": {"entry": 1.0},
        "thought": SimpleNamespace(score=score, direction=direction),
    }


def test_halted_state_produces_visible_vetoed():
    b = _brain()
    b.state.update(policy_state={**DEFAULT_POLICY_STATE, "authorized": False, "halted": True, "pause_reason": "daily_loss_limit"})
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == VETOED
    assert v.reason == "daily_loss_limit"
    assert v.stage == "safety"


def test_valid_edge_admitted_with_size():
    b = _brain()
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == ENTER
    assert v.sizing is not None and v.sizing.shares > 0
    assert v.stop is not None and v.target is not None


def test_direction_vetoed():
    b = _brain()
    b.tick = lambda *a, **k: _fake_result(side="SELL", direction=-1)
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == VETOED and v.reason == "direction_rejected"


def test_no_signal_is_hold():
    b = _brain()
    b.tick = lambda *a, **k: _fake_result(side="HOLD", score=0.5, direction=0)
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == HOLD and v.reason == "no_signal"


def test_governor_cap_vetoes_third():
    b = _brain()
    for i in range(2):
        assert b.decide_entry(f"T{i}", _snap(), {}, "rth").action == ENTER
    v3 = b.decide_entry("T3", _snap(), {}, "rth")
    assert v3.action == VETOED and v3.reason == "cycle_budget"


def test_invalid_snapshot_vetoed():
    b = _brain()
    v = b.decide_entry("NVD", {"prices": [1.0] * 3}, {}, "rth")
    assert v.action == VETOED and v.reason == "no_data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_brain_pipeline.py -v`
Expected: FAIL — `NeuromorphicBrain.decide_entry` missing.

- [ ] **Step 3: Implement**

In `brain/orchestrator.py`:
- import `compute_alpha_from_snap`, `entry_bars` from `hanoon_prime.juli_feed` and `check_tick_latency` from `hanoon_prime.cerebellum` (verify no import cycle: juli_feed imports `brain.indicators`/`brain.regime`/`cerebellum`/`types`, none import orchestrator → safe).
- add `self.governor = Governor()` and `self.probe = ProbeRecovery()` in `__init__` (before `_init_neuromorphic`).
- add `begin_entry_cycle`, `note_entry`, `resume` (clears brain-side `self._pause_until`/`self._paused_for` if present), `set_safety_enabled(enabled)` → `self.state.update(policy_state={**self.state.get("policy_state", DEFAULT_POLICY_STATE), "enabled": enabled})`.
- add `decide_entry` per the gate order above (functions ≤40 lines — split into `_check_snapshot_valid`, `_apply_fast_gates`, `_build_enter` as drafted).
- keep `set_latest_prices` behavior by calling the existing private method with the snapshot.

Implementation guidance (fast-cortex facts, no slow-state access):
- Read all policy from `self.state.get("policy_state", DEFAULT_POLICY_STATE)` only.
- Scale: `scaled = scale_shares(sizing.shares, price, policy_state, sizing)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_brain_pipeline.py -v && mypy src/hanoon_prime/brain/orchestrator.py && python -c "import pathlib; l=pathlib.Path('src/hanoon_prime/brain/orchestrator.py'); print('lines', len(l.read_text().splitlines()))"`
Expected: PASS; mypy clean; orchestrator stays under its R3 line budget (it is skip-listed at 200 but the earlier limit keeps it tidy — Actual length check: only the ≤200 check is exempted; the suite still requires ≤40/func).

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/brain/orchestrator.py tests/test_brain_pipeline.py
git commit -m "feat(brain): decide_entry single decision point with visible veto reasons"
```

### Task 9: `juli.py` — emit Verdicts from the brain

**Files:**
- Modify: `src/hanoon_prime/juli.py`
- Modify: `scripts/smoke_live.py`
- Test: update `tests/test_flow_throttle.py` (rotation now counts `decide_entry` calls)

**Interfaces:**
- `JuliBrain.tick(watch, snapshot, streamer, held_positions, session="rth") -> tuple[list[dict], list[Verdict]]` — same tuple order as today; `session` defaults keep `smoke_live.py` + older tests working.
  - Calls `self.brain.begin_entry_cycle()` once at the top.
  - `_evaluate_entries(ticker, snap, open_positions, session)` → `self.brain.decide_entry(ticker, snap, open_positions, session)` and appends the returned `Verdict` to `self._recent_verdicts` (new `collections.deque(maxlen=200)`) — so `/verdicts` shows the last 200, telemetry last 50.
  - Existing entry-lock throttle: if lock held → fail fast (return `([], [])`) exactly like today.
  - Rotation (`_eval_off`, EVAL_WINDOW): tickers not in the eval window are **not candidates this cycle — scheduling, not a decision**; they produce NO Verdict (documented; this matches the spec's "every *evaluated* ticker" wording). Their stale entries age out of `/verdicts` naturally.
  - `_evaluate_exits` unchanged; returns are merged with `policy_exits` from `self.brain.state.get("policy_exits", [])` so givebacks flow out the exits channel.
- Deletes: `_build_decision`, `_eval_one` (logic lives in `NeuromorphicBrain.decide_entry`).
- `juli.py` must stay ≤200 lines (R3 not skipped): the scoring loop body stays; gate help moved to orchestrator. If tight, extract `_assemble_snapshot`/`_log_decision` into tiny private methods.

- [ ] **Step 1: Update the failing test first**

In `tests/test_flow_throttle.py`, replace the direct `_eval_off`-based rotation assertion with a call-count assertion on a stubbed `brain.decide_entry`:

```python
def test_juli_tick_calls_decide_entry_rotating():
    from hanoon_prime.juli import JuliBrain

    calls = []
    b = JuliBrain(object())
    b.brain.decide_entry = lambda *a, **k: (calls.append(a[0]) or Verdict(ticker=a[0]))
    watch = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
    snap = {t: {"prices": [1.0] * 40, "bid": 1.0, "ask": 1.01, "mid": 1.005, "last": 1.0, "ts": 0.0} for t in watch}
    b.tick(watch, snap, None, set(), session="rth")
    assert calls and len(calls) <= len(watch)
    assert set(calls) <= set(watch)
```

Port the loop/rotation mechanics test to assert `decide_entry` was called for `EVAL_WINDOW` tickers and NOT for the others.

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_flow_throttle.py -v`
Expected: FAIL — `decide_entry` unbound… it will fail because `JuliBrain` has no `decide_entry`-backed tick yet (import/py errors).

- [ ] **Step 3: Rewire juli.py**

```python
def tick(self, watch, snapshot, streamer, held_positions, session="rth"):
    """Brain-first decision loop: returns (exits, verdicts)."""
    if self._lock_held:
        return [], []
    self._lock_held = True
    self.brain.begin_entry_cycle()
    verdicts = []
    try:
        open_positions = self.brain.state.get("positions_open", {})
        for ticker in self._eval_order(watch):
            if ticker in self._eval_off:
                continue
            snap = self._snapshot_for(ticker, snapshot)
            verdicts.append(self.brain.decide_entry(ticker, snap, open_positions, session))
        for v in verdicts:
            self._recent_verdicts.append(v)
        exits = self._evaluate_exits(streamer, held_positions)
        exits += self.brain.state.get("policy_exits", [])
        return exits, verdicts
    finally:
        self._lock_held = False
```

(`_eval_order` = the EVAL_WINDOW rotation index; `_snapshot_for` returns the dict or `None`.) Remove `_build_decision`/`_eval_one` bodies.

- [ ] **Step 4: Update smoke_live.py**

`scripts/smoke_live.py` (line ~191 and the dict-assert region):
```python
verdict = v / d
assert isinstance(verdict, Verdict) or "action" in d
if verdict.action == "ENTER":
    assert verdict.sizing is not None and verdict.sizing.shares >= 0
    assert verdict.regime_canon  # -> resolve: regime lives on verdict.horizon? keep smoke asserting veracity
```
(Adjust the `regime_canon` assertion to the new `Verdict` field ledger — verdicts carry `horizon`, not `regime_canon`; smoke checks `.action`, `.sizing`.)

- [ ] **Step 5: Run the full test sweep**

Run: `pytest tests/test_flow_throttle.py tests/test_neuromorphic.py -v && python -c "import pathlib; print('juli lines', len(pathlib.Path('src/hanoon_prime/juli.py').read_text().splitlines()))"`
Expected: PASS; ρ≤200.

- [ ] **Step 6: Commit**

```bash
git add src/hanoon_prime/juli.py scripts/smoke_live.py tests/test_flow_throttle.py
git commit -m "feat(juli): tick emits Verdicts from decide_entry; rotation intact"
```

### Task 10: `ib_cycle.py` — orchestration only

**Files:**
- Modify: `src/hanoon_prime/ib_cycle.py`
- Test: update `tests/test_bugfixes.py` (TestBug1), `tests/test_flow_throttle.py` (market-closed test stays green)

**Changes (del — all decision logic leaves ib_cycle; add — pure orchestration):**

Delete:
- `_can_trade`, `_entry_throttled`, `_entry_safety_cleared`, `_portfolio_gate_and_size`, `_exec_decision`, `_check_safety`, `_halt`, `_sync_portfolio_risk`
- module singletons `_PORTFOLIO_RISK` / `_PROBE_RECOVERY` (if imported)
- unused `immune` imports + `safety_halt` import (halt notify now lives in `brain/policy/safety.py`)
- `self._halted`, `self._MAX_ENTRIES`, `self._ENTRY_COOLDOWN` attrs if only used by removed helpers

Keep unchanged:
- loop, `_snapshot` context assembly, `sync_from_ib`, `place_bracket`, reconcile, `_attach_position_watchers`, `register_position` (+ brain side), `_reflect_closed`, `_drain_event_exits`, EOD/manual flatten (user commands), `_heartbeat` (temp file + telegram), `_cleanup`, `_SLEEP_MGR` clock (`mkt_state = _SLEEP_MGR.get_state()` — provides the `session` fact), `read_portfolio` (feeds account facts), `count_open_positions`.

Add:
```python
RISK_SYNC_SECS = 30.0

def _publish_account_feed(self, pnl) -> None:
    """Forward IB account facts to the slow cortex on BrainState."""
    daily = float(pnl.dailyPnL) if pnl is not None else 0.0
    feed = {"daily_pnl": daily, "ts": time.monotonic()}
    if time.monotonic() - getattr(self, "_last_policy_sync", 0.0) >= RISK_SYNC_SECS:
        self._last_policy_sync = time.monotonic()
        try:
            summary = self.ib.accountSummary(self.account)
            net_liq = next(
                (float(i.value) for i in summary if i.tag == "NetLiquidation"), 0.0
            )
            feed["equity"] = net_liq
            feed["positions"] = read_portfolio(self.ib)
        except Exception as exc:
            log.debug("Account sync skipped: %s", exc)
    self.juli._state.update(
        account_feed=feed,
        consecutive_losses=getattr(self.hippocampus, "_consecutive_losses", 0),
    )

def _execute_verdict(self, verdict) -> None:
    """Execute one ENTER verdict (market-open gate, brain registration)."""
    ticker = verdict.ticker
    tk = self.streamer.ticker_subs.get(ticker)
    if tk is None or not tk.hasBidAsk or math.isnan(tk.bid) or math.isnan(tk.ask):
        log.warning("VERDICT UNEXECUTABLE %s: no live bid/ask", ticker)
        return
    if ticker in self.hippocampus._open_positions:
        log.info("SKIP %s open", ticker)
        return
    price = float((tk.bid + tk.ask) * 0.5)
    self.executor.place_bracket(
        ticker, verdict.thought, price, self.streamer,
        sizing=verdict.sizing, horizon=verdict.horizon,
    )
    self.executor.last_thoughts[ticker] = verdict.thought
    self.juli.brain.register_position(ticker, price, horizon=verdict.horizon)
    self._attach_position_watchers(ticker)
    self.juli.brain.note_entry(ticker)
```

`_finish_cycle` (reworked):
```python
def _finish_cycle(self, daily_pnl, meta, verdicts):
    """Execute ENTER verdicts (market-open only), journal every verdict."""
    if daily_pnl is not None:
        self.hippocampus._daily_pnl = float(daily_pnl)
    self._publish_account_feed(daily_pnl)
    market_open = bool(meta and meta.market_open)
    for v in verdicts:
        self.journal.append({"event": "verdict", "ts": time.time(), **v.to_dict()})
        if market_open and v.action == ENTER:
            self._execute_verdict(v)
    if meta is not None and market_open:
        self.eod_flatten_or_manual()  # existing user-command flatten
    self._heartbeat()
    self.pnl_chunks_history.app...  # unchanged pnl history append
```

- [ ] **Step 1: Write the failing test** (tests/test_bugfixes.py TestBug1 rewrite)

```python
def test_execute_verdict_places_bracket_and_registers():
    from hanoon_prime.brain.policy.verdict import Verdict, ENTER
    from hanoon_prime import ib_cycle

    bot = _FakeBot()  # existing fixture, minus _exec_decision
    v = Verdict(ticker="T1", action=ENTER, sizing=None, thought=None, horizon="scalp")
    bot.streamer.ticker_subs["T1"] = types.SimpleNamespace(hasBidAsk=True, bid=10.0, ask=10.05)
    bot.hippocampus._open_positions = {}
    bot._execute_verdict(v)  # patched to enqueue instead of place
    assert bot.queued_brackets[0][0] == "T1"
    assert bot.registrations[0] == "T1"
```

(The fake engages `_execute_verdict` with a fake executor whose `place_bracket` records arguments — mirror the current FIX-2026-09-02-01 mechanics; keep an explicit assert that the market-open gate still skips when `meta.market_open` false.)

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_bugfixes.py -v`
Expected: FAIL — `_execute_verdict` missing / signature change breaks old TestBug1.

- [ ] **Step 3: Apply the ib_cycle rewrite**

- [ ] **Step 4: Sweep — full suite + format check**

Run: `pytest tests/test_flow_throttle.py tests/test_bugfixes.py tests/test_telemetry.py tests/test_brain_pipeline.py -v && pre-commit run --files src/hanoon_prime/ib_cycle.py` (dry note: `ib_cycle.py` is black-excluded; the hook should skip it — never let black reformat it).

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/ib_cycle.py tests/test_bugfixes.py
git commit -m "refactor(ib_cycle): orchestration only — verdicts drive execution"
```

### Task 11: Telemetry re-points to policy_state + `/verdicts`

**Files:**
- Modify: `src/hanoon_prime/telemetry.py`
- Test: update `tests/test_telemetry.py`

**Changes:**
- `/health` → read `juli.brain.state.get("policy_state", {})` for `("halted", False)`, `("authorized", True)`, `("enabled", True)` + `_heartbeat`; wrap in `hasattr(bot, "juli")` guard (tests' `_FakeBot` has no JuliBrain) — fall back to defaults on missing policy_state.
- `/risk` → policy_state subset (`equity", "drawdown", "exposure", "stress_mode", "risk_scalar", "position_count", "max_positions", "daily_pnl", "consecutive_losses"`).
- `/safety-net`:
  - GET → `{enabled, halted, authorized, pause_reason}` from policy_state.
  - POST → `bot.juli.brain.resume()` on `"action": "resume"`; `bot.juli.brain.set_safety_enabled(bool)` on `"action": "toggle"`; guard with `hasattr`.
- `/config` → GET/POST SURFACE unchanged (still `TRADING_CONFIG`), now sourced from the shim (Task 3 already keeps the object identical).
- NEW `/verdicts` route:
```python
@app.get("/verdicts")
def verdicts():
    recent = []
    if hasattr(bot, "juli") and getattr(bot.juli, "_recent_verdicts", None):
        recent = [v.to_dict() for v in list(bot.juli._recent_verdicts)[-50:]]
    return {"count": len(recent), "verdicts": recent}
```
- Telemetry stays an OBSERVER for decision state; POSTs are user commands (brain methods, not reporter mutations).

- [ ] **Step 1: Write the failing test**

In `tests/test_telemetry.py`, extend fixtures: give `_FakeBot` a `juli`-less guard path and assert:
```python
def test_health_reads_policy_state_when_present(client, monkeypatch):
    monkeypatch.setattr(bot, "juli", simple(brain=simple(state=StateUnderTest({...halted...}))))
    r = client.get("/health")
    assert r.json()["status"]["halted"] is True

def test_halt_visible_in_health_and_resume_command_called(client):
    r = client.post("/safety-net", json={"action": "resume"})
    assert r.status_code == 200
    assert bot.juli.brain.resume_called

def test_verdicts_endpoint_lists_recent(client, monkeypatch):
    ...assert GET /verdicts returns the 1 seeded Verdict with action+reason...
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_telemetry.py -v`
Expected: FAIL — routes/attr guards missing.

- [ ] **Step 3: Implement telemetry changes per above; keep `filterwarnings` clean** (no leftover `bot._halted` reads without fallback).

- [ ] **Step 4: Sweep**

Run: `pytest tests/test_telemetry.py -v && mypy src/hanoon_prime/telemetry.py && pre-commit run --files src/hanoon_prime/telemetry.py` and confirm the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/hanoon_prime/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): policy_state-backed health/risk/safety + /verdicts route"
```

### Task 12: Wiring, docs, final verification

**Files:**
- Modify: `src/hanoon_prime/ib_adapter.py`, `docs/ARCHITECTURE.md`

- [ ] **Step 1: Journal wiring**

In `ib_adapter.py`, after `self.juli = JuliBrain(self.ib)`:
```python
consolidation = getattr(self.juli.brain, "_consolidation", None)
if consolidation is not None:
    consolidation.safety.attach_journal(self.journal)
```
(guard: `_consolidation` may be None when neuromorphic disabled).

- [ ] **Step 2: ARCHITECTURE.md pipeline update**

Rewrite the decision section: JULI → `NeuromorphicBrain.decide_entry` gates (validity→signal→trading_policy→safety/probe→governor→sizing→portfolio→scale→ENTER); slow cortex `_update_policy` owns portfolio/safety state and publishes `policy_state`/`policy_exits`; `ib_cycle` = orchestrator + `_execute_verdict` + account feed. Note the two Hippocampus instances and that `hands.py` remains a sim harness.

- [ ] **Step 3: Full verification matrix (EVIDENCE, not assertion)**

Run:
```bash
pytest -q
mypy src/hanoon_prime
pre-commit run --all-files
python -c "import pathlib
for f in sorted(pathlib.Path('src/hanoon_prime/brain/policy').glob('*.py')):
    print(f, len(f.read_text().splitlines()))"
python -c "import pathlib; print('juli', len(pathlib.Path('src/hanoon_prime/juli.py').read_text().splitlines()))"
```
Expected: full suite green, mypy clean, black/isort pass (ib_cycle untouched), every policy file ≤200, juli ≤200, test_contract.py passes unedited.

- [ ] **Step 4: Final commit & live-bot note**

Commit any residual churn. Do NOT restart the live bot yourself — present the diff and coordinate the restart (PID 21141, port 8080, NDRA open) with the user, then verify one full clean cycle via `/verdicts` + journal `{"event":"verdict"}` lines.

## Post-Plan Verification (execution-time)

- Live sanity: `curl localhost:8080/verdicts | python -m json.tool` shows per-ticker ENTER/HOLD/VETOED with reasons; `curl localhost:8080/health | jq .status` no false halt; journal shows `verdict` events each cycle.
- Regression evidence: backfill the observed silent block — with `policy_state.authorized=False` injected, `/verdicts` shows `VETOED` with a reason (the whole reason this plan exists).
- Rollback: `git checkout ace4da7 -- src/hanoon_prime` + restart; the old `ib_cycle` gates are still fully intact in git history.
- Known tradeoff (accepted in spec): halt latency moves from every-main-cycle to slow-cortex cadence (~30s).