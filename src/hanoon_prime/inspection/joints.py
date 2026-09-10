"""Joint registry — run_all produces the full manifest."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .checks import FAIL, MANIFEST_STATUS, OK, WARN, CheckResult, CheckSpec, run_check
from .ctx import InspectionContext
from .halim import halim_engaged_in_decisions, halim_state_matches_clock
from .journals import (
    chain_intact_from_anchor,
    journal_grows,
    seq_forward,
    verdicts_valid,
)
from .lifecycle import (
    lifecycle_execution,
    lifecycle_learning,
    lifecycle_scan,
    lifecycle_seed,
    lifecycle_stream,
    lifecycle_verdict,
)
from .live import bars_advance_when_active, positions_marked_live
from .oracle import closes_reconciled, enters_minted, equity_synced
from .purity import brain_fields_bounded, no_test_episodes, weights_finite_in_band
from .runtime import (
    cycle_flows_when_active,
    health_ok,
    heartbeat_fresh,
    positions_reconciled,
    positions_surface,
    send_healthy,
    sleep_is_expected,
    snapshot_fresh,
    state_matches_clock,
    telegram_configured,
)
from .safety import (
    drawdown_bound,
    no_error_burst,
    no_learn_blocked,
    no_netting_guard,
    no_safety_halt,
    no_traceback,
    policy_flags,
)
from .system import (
    bot_alive,
    bot_from_trusted_checkout,
    cloudflared_alive,
    gateway_watchdog_alive,
    halim_alive,
    monitor_alive,
    single_bot,
)
from .trade_quality import (
    halim_postmortem,
    pnl_sign_consistency,
    profit_factor,
    win_rate,
)
from .weight_purity import cortex_score_degenerate, regime_weights_bounded

JOINT_ORDER = [
    "processes",
    "identity",
    "telemetry",
    "pipeline",
    "session",
    "lifecycle",
    "safety",
    "memory",
    "purity",
    "execution_oracle",
    "halim",
    "trade_quality",
    "notify",
]

SPECS: tuple[CheckSpec, ...] = (
    CheckSpec("processes", "bot_alive", bot_alive, report=True),
    CheckSpec("processes", "halim_alive", halim_alive, report=True),
    CheckSpec("processes", "monitor_alive", monitor_alive, report=True),
    CheckSpec("processes", "cloudflared_alive", cloudflared_alive, report=True),
    CheckSpec(
        "processes", "gateway_watchdog_alive", gateway_watchdog_alive, report=True
    ),
    CheckSpec("processes", "single_bot", single_bot, report=True),
    CheckSpec(
        "identity", "bot_from_trusted_checkout", bot_from_trusted_checkout, hard=True
    ),
    CheckSpec("telemetry", "health_ok", health_ok, hard=True),
    CheckSpec("telemetry", "snapshot_fresh", snapshot_fresh, hard=True),
    CheckSpec("telemetry", "positions_surface", positions_surface, report=True),
    CheckSpec("telemetry", "positions_marked_live", positions_marked_live, hard=True),
    CheckSpec("pipeline", "heartbeat_fresh", heartbeat_fresh, hard=True),
    CheckSpec(
        "pipeline", "cycle_flows_when_active", cycle_flows_when_active, hard=True
    ),
    CheckSpec(
        "pipeline", "bars_advance_when_active", bars_advance_when_active, hard=True
    ),
    CheckSpec("pipeline", "sleep_is_expected", sleep_is_expected, report=True),
    CheckSpec("session", "state_matches_clock", state_matches_clock, report=True),
    CheckSpec("session", "positions_reconciled", positions_reconciled, hard=True),
    CheckSpec("lifecycle", "stage_seed", lifecycle_seed, report=True),
    CheckSpec("lifecycle", "stage_stream", lifecycle_stream, report=True),
    CheckSpec("lifecycle", "stage_scan", lifecycle_scan, report=True),
    CheckSpec("lifecycle", "stage_verdict", lifecycle_verdict, report=True),
    CheckSpec("lifecycle", "stage_execution", lifecycle_execution, report=True),
    CheckSpec("lifecycle", "stage_learning", lifecycle_learning, report=True),
    CheckSpec("safety", "no_netting_guard", no_netting_guard, hard=True),
    CheckSpec("safety", "no_traceback", no_traceback, hard=True),
    CheckSpec("safety", "no_safety_halt", no_safety_halt, hard=True),
    CheckSpec("safety", "no_learn_blocked", no_learn_blocked, report=True),
    CheckSpec("safety", "policy_flags", policy_flags, report=True),
    CheckSpec("safety", "drawdown_bound", drawdown_bound, hard=True),
    CheckSpec("safety", "no_error_burst", no_error_burst, hard=True),
    CheckSpec("memory", "journal_grows", journal_grows, hard=True),
    CheckSpec("memory", "seq_forward", seq_forward, report=True),
    CheckSpec("memory", "verdicts_valid", verdicts_valid, hard=True),
    CheckSpec(
        "memory", "chain_intact_from_anchor", chain_intact_from_anchor, hard=True
    ),
    CheckSpec("purity", "no_test_episodes", no_test_episodes, hard=True),
    CheckSpec("purity", "weights_finite_in_band", weights_finite_in_band, hard=True),
    CheckSpec("purity", "regime_weights_bounded", regime_weights_bounded, hard=True),
    CheckSpec(
        "purity", "cortex_score_degenerate", cortex_score_degenerate, report=True
    ),
    CheckSpec("purity", "brain_fields_bounded", brain_fields_bounded, hard=True),
    CheckSpec("execution_oracle", "enters_minted", enters_minted, report=True),
    CheckSpec("execution_oracle", "closes_reconciled", closes_reconciled, report=True),
    CheckSpec("execution_oracle", "equity_synced", equity_synced, report=True),
    CheckSpec(
        "halim", "halim_state_matches_clock", halim_state_matches_clock, report=True
    ),
    CheckSpec("notify", "telegram_configured", telegram_configured, report=True),
    CheckSpec("notify", "send_healthy", send_healthy, report=True),
    CheckSpec("trade_quality", "win_rate", win_rate, report=True),
    CheckSpec("trade_quality", "profit_factor", profit_factor, report=True),
    CheckSpec("trade_quality", "halim_postmortem", halim_postmortem, report=True),
    CheckSpec(
        "trade_quality", "pnl_sign_consistency", pnl_sign_consistency, report=True
    ),
    CheckSpec(
        "halim", "halim_engaged_in_decisions", halim_engaged_in_decisions, report=True
    ),
)

HARD_KEYS: frozenset[tuple[str, str]] = frozenset(
    (s.joint, s.name) for s in SPECS if s.hard
)
REPORT_KEYS: frozenset[tuple[str, str]] = frozenset(
    (s.joint, s.name) for s in SPECS if s.report
)


@dataclass(frozen=True)
class Manifest:
    """One probe tick: full check results plus provenance."""

    ts: float
    git_head: str | None
    pid: int
    results: tuple[CheckResult, ...]

    @property
    def status(self) -> str:
        """Aggregate MANIFEST_STATUS over the hard/report spec keys."""
        return MANIFEST_STATUS(list(self.results), HARD_KEYS)

    @property
    def hard_fails(self) -> list[CheckResult]:
        """Hard-spec checks currently FAILing."""
        return [
            r
            for r in self.results
            if (r.joint, r.name) in HARD_KEYS and r.status == FAIL
        ]

    @property
    def anomalies(self) -> list[CheckResult]:
        """Reported checks in WARN or FAIL."""
        return [
            r
            for r in self.results
            if (r.joint, r.name) in REPORT_KEYS and r.status in (WARN, FAIL)
        ]


def run_all(ctx: InspectionContext) -> Manifest:
    """Execute every registered check against a single tick context."""
    results = tuple(run_check(s, ctx) for s in SPECS)
    return Manifest(
        ts=time.time(), git_head=ctx.git_head(), pid=os.getpid(), results=results
    )
