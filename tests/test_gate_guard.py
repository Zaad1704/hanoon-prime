"""tests/test_gate_guard.py — fail-closed rollout-gate guard (R30 runtime arm).

Proves gate_guard detects and force-resets mid-process read-site drift at the
decision boundary, tolerates a legitimate (evidence-backed) promotion, and that
the live brain snapshots its declared gates at start().
"""

from __future__ import annotations

import hanoon_prime.brain.orchestrator as orb
from hanoon_prime import immune
from hanoon_prime.brain.gate_guard import (
    declared_gates,
    drift,
    enforce,
    verify_decision_boundary,
)
from hanoon_prime.brain.orchestrator import NeuromorphicBrain


def test_declared_gates_are_all_off_by_default():
    gates = declared_gates()
    assert len(gates) == 10
    assert all(
        value is False for value in gates.values()
    ), f"all gates must start OFF: {gates}"


def test_drift_is_detected_and_force_reset(monkeypatch):
    monkeypatch.setattr(orb, "NEURO_BLEND_ENABLED", True)
    declared = declared_gates()
    assert drift(declared) == {"NEURO_BLEND_ENABLED": True}
    assert enforce(declared) is False
    assert orb.NEURO_BLEND_ENABLED is False
    assert verify_decision_boundary(declared) is True


def test_promoted_declaration_is_not_reset(monkeypatch):
    monkeypatch.setattr(immune, "NEURO_BLEND_ENABLED", True)
    monkeypatch.setattr(orb, "NEURO_BLEND_ENABLED", True)
    declared = declared_gates()
    assert enforce(declared) is True
    assert orb.NEURO_BLEND_ENABLED is True


def test_decision_boundary_is_a_noop_before_start():
    assert verify_decision_boundary(None) is True


def test_start_snapshots_declared_gates():
    brain = NeuromorphicBrain(persist_memory=False)
    assert brain._gate_declared is None
    brain.start()
    assert brain._gate_declared == declared_gates()
