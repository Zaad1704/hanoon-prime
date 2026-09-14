"""brain.halim_evidence_prompt — HALIM's evidence-grounded CoT prompt.

Split out of :mod:`halim_evidence` to honour the R3 200-line file contract.
The prompt grounds HALIM's self-improvement in EVAL evidence and in Juli's
win/loss pillar, which must be held upright.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_evidence_prompt"]


def build_evidence_prompt(evidence: dict[str, Any]) -> str:
    """Build a chain-of-thought prompt grounding HALIM's self-improvement."""
    return (
        "You are HALIM, the trading architect for Juli (a live trading brain).\n"
        "You just closed a learning cycle. Analyze the EVAL evidence below and "
        "reason step by step (chain of thought) about WHICH gate strangles "
        "entries and whether the cortex had real conviction behind the "
        "rejections. Then recommend bounded parameter adjustments to make Juli "
        "better. Also keep Juli's PILLAR upright: the pillar is the win/loss "
        "edge against break-even (win_rate - 1/(1+avg_win/avg_loss)); upright "
        "(edge >= 0) means Juli is winning, and any lean is failure regardless "
        "of side. Recommend changes that correct a leaning pillar toward "
        "upright. Return EXACTLY this JSON, nothing else:\n"
        '{"reasoning": "<your chain of thought: which gate dominates, why the '
        'cortex is right/wrong, how the pillar is leaning, what to adjust>", '
        '"recommendations": ['
        '{"action": "adjust_threshold", "param": "threshold", "value": <0.40-0.80>, '
        '"reason": "<why>"}, {"action": "adjust_weight", "param": "<name>", '
        '"value": <-2.0-2.0>, "reason": "<why>"}]}\n'
        "EVIDENCE:\n"
        f"- regime: {evidence['regime']}\n"
        f"- eval_lines_checked: {evidence['eval_lines']}\n"
        f"- total_vetoes: {evidence['vetoes']}\n"
        f"- vetoes_by_reason: {evidence['by_reason']}\n"
        f"- top_veto_reason: {evidence['top_veto_reason']}\n"
        f"- direction_rejected_count: {evidence['direction_rejected']}\n"
        f"- direction_rejected_mean_abs_score: "
        f"{evidence['direction_rejected_mean_abs_score']}\n"
        f"- real_conviction_discarded: {evidence['real_conviction_discarded']}\n"
        f"- win_rate: {evidence['win_rate']}\n"
        f"- worst_conf_bin_losses: {evidence['worst_conf_bin_losses']}\n"
        f"- pillar_state: {evidence['pillar_state']}\n"
        f"- pillar_edge: {evidence['pillar_edge']}\n"
        f"- pillar_tilt: {evidence['pillar_tilt']}\n"
        f"- win_loss_record: {evidence['win_loss_record']}\n"
        f"- win_loss_net_pct: {evidence['win_loss_net']}\n"
    )
