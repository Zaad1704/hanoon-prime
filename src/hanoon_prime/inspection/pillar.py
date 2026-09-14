"""hanoon_prime.inspection.pillar — the win/loss pillar (edge vs break-even).

The pillar stands at 90 degrees when Juli's realized win rate clears the
break-even rate implied by its payoff ratio (``avg_win / avg_loss``). Any
negative edge is a lean, and a lean is failure regardless of which side
(long or short) produced it — the RL cart-pole objective applied to trading.

The brain publishes the identical awareness shape from
``ConsolidationEngine`` into ``brain_state.pillar``; the juli webapp see-saw
renders it. Directional conviction/veto geometry is retained as secondary
evidence only (it no longer drives the pillar's status).
"""

from __future__ import annotations

from typing import Any

from ..brain.pillar_awareness import compute_pillar_awareness, win_loss_record
from ..brain.policy.trading_policy import TRADING_CONFIG
from ..immune import PILLAR_EDGE_FALL
from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .pillar_evidence import _full_evidence, _parse_eval_lines, _warm_evidence
from .probe import runtime_state, snapshot

# Win/loss pillar states -> Inside Man verdicts.
_STATE_STATUS: dict[str, str] = {
    "upright": OK,
    "warming": OK,
    "tipping": WARN,
    "fallen": FAIL,
}


def _cr(name: str, status: str, detail: str, **ev: Any) -> CheckResult:
    """Build an inside_man CheckResult."""
    return CheckResult("inside_man", name, status, detail, dict(ev) if ev else {})


def _read_pillar(ctx: InspectionContext) -> dict[str, Any] | None:
    """Live pillar awareness from brain_state, else derive from realized store."""
    state = runtime_state(ctx)
    brain_state = state.get("brain_state", {})
    if isinstance(brain_state, dict):
        live = brain_state.get("pillar")
        if isinstance(live, dict) and live:
            return live
    realized = state.get("realized")
    if isinstance(realized, dict):
        samples = realized.get("rr_samples", [])
        return compute_pillar_awareness(win_loss_record(samples))
    return None


def _detail(pil: dict[str, Any]) -> str:
    """Human-readable pillar posture from the awareness shape."""
    record = pil.get("record", "0W-0L")
    edge = float(pil.get("edge", 0.0) or 0.0)
    state = str(pil.get("state", "warming"))
    if state == "upright":
        return f"pillar upright — edge {edge:+.3f} ({record})"
    if state == "warming":
        return f"pillar warming up ({record})"
    if state == "tipping":
        return f"pillar tipping — edge {edge:+.3f} ({record})"
    return f"pillar fallen — edge {edge:+.3f} ({record})"


def pillar_balance(ctx: InspectionContext) -> CheckResult:
    """Win/loss edge pillar — upright means winning, any lean means failing."""
    lc, sc, vl, vs, el, pvl, pvs = _parse_eval_lines(ctx)
    pil = _read_pillar(ctx)
    if pil is None or int(pil.get("trades", 0) or 0) == 0:
        return _cr(
            "pillar_balance",
            OK,
            "warming up (no closed trades yet)",
            **_warm_evidence(el),
        )
    total = lc + sc
    ratio = abs(lc - sc) / total if total > 0 else 0.0
    mx, mn = (vl, vs) if vl >= vs else (vs, vl)
    skew = mx / mn if mn > 0 else float(mx)
    bs = runtime_state(ctx).get("brain_state", {})
    hm = bs.get("halim_modifier", 0.0) if isinstance(bs, dict) else 0.0
    cfg = snapshot(ctx).get("config", {})
    mode = cfg.get("direction_mode", TRADING_CONFIG.direction_mode)
    status = _STATE_STATUS.get(str(pil.get("state")), WARN)
    ev = _full_evidence(lc, sc, vl, vs, pvl, pvs, ratio, skew, hm, el)
    ev.update(
        {
            "direction_mode": mode,
            "pillar_state": pil.get("state"),
            "upright": bool(pil.get("upright", False)),
            "tilt": pil.get("tilt", 0.0),
            "edge": pil.get("edge", 0.0),
            "win_rate": pil.get("win_rate", 0.0),
            "break_even_wr": pil.get("break_even_wr", 0.0),
            "wins": pil.get("wins", 0),
            "losses": pil.get("losses", 0),
            "trades": pil.get("trades", 0),
            "win_loss_record": pil.get("record", "0W-0L"),
            "net_pnl": pil.get("net", 0.0),
            "r_r": pil.get("r_r", 0.0),
            "edge_bands": [0.0, PILLAR_EDGE_FALL],
        }
    )
    return _cr("pillar_balance", status, _detail(pil), **ev)


__all__ = ["pillar_balance"]
