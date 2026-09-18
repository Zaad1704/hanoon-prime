"""hanoon_prime.brain.gate_guard — fail-closed enforcement of rollout gates.

R29 pins the 10 rollout gates OFF in immune.py; R30 proves the live read
sites are OFF in a pristine process. This guard is the runtime arm: it
snapshots the DECLARED gate values at brain start and, at the decision
boundary, verifies every read-site still matches its declaration.

Lazy call-time readers (bridge.apply_learning, adaptive_thresholds_wiring.
feed_market_env, moe_gate.route_experts) re-import <<immune>> on every call,
so they equal the declaration by construction and cannot drift; only the
at-import module aliases (NEURO_BLEND_ENABLED in orchestrator, etc.) can.
When a drift is detected the guard logs an ERROR and force-resets the
drifted aliases to their declared value — the loop keeps running rollout-OFF
(fail-closed = inert, not crash).

The guard is promotion-aware: enforcing DECLARED, not OFF. A legitimate
promotion flips immune.py AND propagates the read sites in a fresh process;
declared == read -> no reset, the organ runs.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Tuple

from .. import immune

log = logging.getLogger("hanoon_prime.gate_guard")

GATE_NAMES: Tuple[str, ...] = (
    "CALIBRATION_NUDGE_ENABLED",
    "HYSTERESIS_EXIT_ENABLED",
    "PROBE_RECOVERY_ENABLED",
    "CONTRARIAN_MODE_ENABLED",
    "DELIBERATION_TRACE_ENABLED",
    "HALIM_EVIDENCE_LEARNING",
    "NEURO_BLEND_ENABLED",
    "NEURO_LEARN_ENABLED",
    "NEURO_ADAPTIVE_THRESHOLD_ENABLED",
    "NEURO_MOE_GATE_ENABLED",
)

# gate name -> ((module path, attr), ...) read sites that bind at import time.
_ALIAS_SITES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "CALIBRATION_NUDGE_ENABLED": (
        ("hanoon_prime.brain.realized_ev", "CALIBRATION_NUDGE_ENABLED"),
    ),
    "HYSTERESIS_EXIT_ENABLED": (
        ("hanoon_prime.brain.exit_ladder", "HYSTERESIS_EXIT_ENABLED"),
    ),
    "PROBE_RECOVERY_ENABLED": (
        ("hanoon_prime.brain.probe_recovery", "PROBE_RECOVERY_ENABLED"),
    ),
    "CONTRARIAN_MODE_ENABLED": (
        ("hanoon_prime.contrarian", "CONTRARIAN_MODE_ENABLED"),
    ),
    "DELIBERATION_TRACE_ENABLED": (
        ("hanoon_prime.brain.orchestrator", "DELIBERATION_TRACE_ENABLED"),
    ),
    "HALIM_EVIDENCE_LEARNING": (
        ("hanoon_prime.inspection.pillar_evidence", "HALIM_EVIDENCE_LEARNING"),
        ("hanoon_prime.brain.consolidation", "HALIM_EVIDENCE_LEARNING"),
    ),
    "NEURO_BLEND_ENABLED": (
        ("hanoon_prime.brain.orchestrator", "NEURO_BLEND_ENABLED"),
    ),
    # NEURO_LEARN / ADAPTIVE_THRESHOLD / MOE are lazy call-time readers only.
}


def declared_gates() -> Dict[str, bool]:
    """The canonical declaration — current immune.py literals."""
    return {name: bool(getattr(immune, name, False)) for name in GATE_NAMES}


def drift(declared: Dict[str, bool]) -> Dict[str, bool]:
    """Return gate→value pairs where a live read-site disagrees with declared."""
    drifted: Dict[str, bool] = {}
    for gate, expected in declared.items():
        for module_name, attr in _ALIAS_SITES.get(gate, ()):
            value = bool(getattr(importlib.import_module(module_name), attr))
            if value is not expected:
                drifted.setdefault(gate, value)
    return drifted


def enforce(declared: Dict[str, bool]) -> bool:
    """Fail-closed check for the decision boundary.

    Returns True when every read-site matches the declaration. On drift the
    offending aliases are reset to declared and False is returned — an organ
    can never silently activate mid-process.
    """
    drifted = drift(declared)
    if not drifted:
        return True
    log.error(
        "gate_guard: gate drift detected at decision boundary %s — "
        "force-resetting to declared; live path continues rollout-OFF",
        drifted,
    )
    for gate in drifted:
        expected = declared[gate]
        for module_name, attr in _ALIAS_SITES.get(gate, ()):
            setattr(importlib.import_module(module_name), attr, expected)
    return False


def verify_decision_boundary(declared: Optional[Dict[str, bool]]) -> bool:
    """One-shot guard for the decision loop (a no-op without a snapshot)."""
    if declared is None:
        return True
    return enforce(declared)
