"""hanoon_prime.brain.telemetry_facade — Brain telemetry facade.

Decouples the NeuromorphicBrain god object from its telemetry serialization:
orchestrator.snapshot() delegates here, so the dict contract (its keys) lives
in exactly one place and stays stable across refactors, and the object no
longer owns the rendering of its own state.

Behaviour-identical extraction: reads the same attributes in the same order
and mirrors the same None-guards (sleep engine). The brain argument is typed
loose on purpose — the facade is the boundary between the brain internals and
the telemetry surface, and wants no new coupling.
"""

from __future__ import annotations

from typing import Any

from .telemetry_summaries import extinction_summary

#: The stable key contract for the brain snapshot. Any consumer of the
#: telemetry JSON depends on these names — do not rename without migration.
SNAPSHOT_KEYS: tuple[str, ...] = (
    "memory",
    "realized",
    "episodic_size",
    "extinction_size",
    "extinction",
    "metacog",
    "threshold",
    "brain_state",
    "decision_count",
    "neuromorphic",
    "nash",
    "advisor",
    "exits_adaptive",
    "meta_label",
    "meta_label_dnn",
    "horizon_bandit",
    "regime_weights",
    "learned_exit",
    "genome",
    "strategy_research",
    "sleep_engine",
)


def build_brain_snapshot(brain: Any) -> dict[str, Any]:
    """Full brain snapshot for telemetry (see :data:`SNAPSHOT_KEYS`)."""
    result: dict[str, Any] = {
        "memory": brain.memory.snapshot(),
        "realized": brain._realized.snapshot(),
        "episodic_size": brain.episodic.size,
        "extinction_size": brain._extinction.size,
        "extinction": extinction_summary(brain._extinction.save().get("cells", [])),
        "metacog": brain._meta_cog.snapshot(),
        "threshold": brain.dynamics.threshold,
        "brain_state": brain.state.snapshot(),
        "decision_count": brain._decision_count,
        "neuromorphic": (brain._neuromorphic.snapshot() if brain._neuromorphic else {}),
        "nash": brain.nash.get_telemetry(),
        "advisor": brain._advisor.snapshot(),
        "exits_adaptive": brain.exits.telemetry(),
        "meta_label": brain._meta.snapshot(),
        "meta_label_dnn": brain._meta.dnn_snapshot(),
        "horizon_bandit": brain._bandit.snapshot(),
        "regime_weights": brain._regime_weights.snapshot(),
        "learned_exit": {"trades": brain._learned_exit_trade_count()},
        "genome": brain.genome.get_genome(),
        "strategy_research": {
            "registry": brain.strategy_registry.snapshot(),
            "bandit": brain.strategy_bandit.snapshot(),
            "research": brain.strategy_research.snapshot(),
            "in_play": dict(brain._last_strategy),
            "shadow_book": brain._shadow_book.snapshot(),
        },
    }
    if getattr(brain, "_sleep_engine", None) is not None:
        result["sleep_engine"] = brain._sleep_engine.snapshot()
    return result


__all__ = ["SNAPSHOT_KEYS", "build_brain_snapshot"]
