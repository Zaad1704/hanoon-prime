"""Fresh-interpreter runtime audit: the 10 rollout gates OFF at every read-site.

Spawned in a pristine process (like the live bot), imports every consumer
module so their at-import aliases bind, constructs the production brain
exactly as ib_adapter does, and dumps the values each live decision path
actually branches on.

Read-site families:
  * immune.<FLAG>            - the canonical literal (pinned by R29)
  * <consumer module>.FLAG   - module-aliases bound at import (NOT pinned by R29;
                               this audit exists because those can drift from immune)
  * lazy from ...immune import  - bridge / adaptive_thresholds_wiring / moe_gate
                               re-read immune at call time (== immune literal)
"""

import json
import sys

from hanoon_prime import contrarian
from hanoon_prime import immune as imm
from hanoon_prime.brain import consolidation
from hanoon_prime.brain import orchestrator as orb
from hanoon_prime.brain import probe_recovery, realized_ev
from hanoon_prime.brain.exit_ladder import HYSTERESIS_EXIT_ENABLED
from hanoon_prime.brain.neurons import adaptive_thresholds_wiring, bridge, moe_gate
from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.shared_state import BrainState
from hanoon_prime.inspection import pillar_evidence

brain = NeuromorphicBrain(brain_state=BrainState(), persist_memory=False)
brain.start()

sites = {
    "immune.CALIBRATION_NUDGE_ENABLED": imm.CALIBRATION_NUDGE_ENABLED,
    "immune.HYSTERESIS_EXIT_ENABLED": imm.HYSTERESIS_EXIT_ENABLED,
    "immune.PROBE_RECOVERY_ENABLED": imm.PROBE_RECOVERY_ENABLED,
    "immune.CONTRARIAN_MODE_ENABLED": imm.CONTRARIAN_MODE_ENABLED,
    "immune.DELIBERATION_TRACE_ENABLED": imm.DELIBERATION_TRACE_ENABLED,
    "immune.HALIM_EVIDENCE_LEARNING": imm.HALIM_EVIDENCE_LEARNING,
    "immune.NEURO_BLEND_ENABLED": imm.NEURO_BLEND_ENABLED,
    "immune.NEURO_LEARN_ENABLED": imm.NEURO_LEARN_ENABLED,
    "immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED": imm.NEURO_ADAPTIVE_THRESHOLD_ENABLED,
    "immune.NEURO_MOE_GATE_ENABLED": imm.NEURO_MOE_GATE_ENABLED,
    # at-import aliases the live code actually branches on:
    "orchestrator.NEURO_BLEND_ENABLED": orb.NEURO_BLEND_ENABLED,
    "orchestrator.DELIBERATION_TRACE_ENABLED": orb.DELIBERATION_TRACE_ENABLED,
    "exit_ladder.HYSTERESIS_EXIT_ENABLED": HYSTERESIS_EXIT_ENABLED,
    "realized_ev.CALIBRATION_NUDGE_ENABLED": realized_ev.CALIBRATION_NUDGE_ENABLED,
    "probe_recovery.PROBE_RECOVERY_ENABLED": probe_recovery.PROBE_RECOVERY_ENABLED,
    "contrarian.CONTRARIAN_MODE_ENABLED": contrarian.CONTRARIAN_MODE_ENABLED,
    "pillar_evidence.HALIM_EVIDENCE_LEARNING": pillar_evidence.HALIM_EVIDENCE_LEARNING,
    "consolidation.HALIM_EVIDENCE_LEARNING": consolidation.HALIM_EVIDENCE_LEARNING,
    "moe_gate.route_experts gated (immune)": imm.NEURO_MOE_GATE_ENABLED,
    "adaptive_thresholds_wiring gated (immune)": imm.NEURO_ADAPTIVE_THRESHOLD_ENABLED,
    "bridge.apply_learning gated (immune)": imm.NEURO_LEARN_ENABLED,
}

print(
    "diagnostics:",
    json.dumps(
        {
            "neuromorphic_engine_constructed": brain._neuromorphic is not None,
            "sleep_engine_on_market_close": brain._sleep_engine is not None,
            "C1_replay_plasticity_gated_by_learn": imm.NEURO_LEARN_ENABLED,
        },
        sort_keys=True,
    ),
)
print(json.dumps(sites, indent=2, sort_keys=True))
bad = [k for k, v in sites.items() if v is not False]
print(f"AUDIT_RESULT={'FAIL' if bad else 'PASS'}")
sys.exit(1 if bad else 0)
