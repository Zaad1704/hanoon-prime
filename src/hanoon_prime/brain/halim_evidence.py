"""hanoon_prime.brain.halim_evidence — evidence-grounded HALIM CoT learning.

Feeds the 'why not winning' EVAL evidence to HALIM's /v1/complete CoT;
recs route through the existing bounded applier (bounds + 300s cooldown
+ max 3). Byte-identical until HALIM_EVIDENCE_LEARNING is enabled.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections import deque
from typing import Any, Callable

from .config import STATE_DIR
from .halim_adapter import _normalize_halim_json
from .halim_recommendations import validate_recommendation

log = logging.getLogger(__name__)

EVAL_LOG_PATH = STATE_DIR.parent / "logs" / "hanoon_prime.log"
EVAL_MARKER = re.compile(r"EVAL\s+[A-Z]")
# TICKER:ACTION(score,side)[stage:reason]
_VERDICT_RE = re.compile(
    r"(?P<ticker>[A-Z0-9.]+):(?P<action>VETOED|HOLD|BUY|SELL|PRESSED|ENTER)"
    r"\((?P<score>-?\d+\.\d+),[^)]*\)\[(?P<stage>[^:\]]*):(?P<reason>[^\]]*)\]"
)
_REAL_CONVICTION: float = 0.5
_WINDOW: int = 1000


def read_eval_tail(path: str, n: int = _WINDOW) -> list[str]:
    """Read the last *n* non-empty lines of the EVAL log (best-effort)."""
    try:
        with open(path, "r", errors="replace") as fh:
            tail = deque(fh, maxlen=n)
        return [ln for ln in tail if ln.strip()]
    except (FileNotFoundError, OSError):
        return []


def collect_evidence(
    log_lines: list[str],
    realized_snapshot: dict[str, Any] | None = None,
    regime: str = "unknown",
    win_rate: float | None = None,
) -> dict[str, Any]:
    """Tally recent vetoes + direction_rejected conviction + conf-bin losses."""
    by_reason: dict[str, int] = {}
    dir_scores: list[float] = []
    eval_lines = 0
    for line in log_lines[-_WINDOW:]:
        if not EVAL_MARKER.search(line):
            continue
        eval_lines += 1
        for m in _VERDICT_RE.finditer(line):
            if m.group("action") != "VETOED":
                continue
            reason = m.group("reason")
            by_reason[reason] = by_reason.get(reason, 0) + 1
            if reason == "direction_rejected":
                dir_scores.append(abs(float(m.group("score"))))
    dir_n = len(dir_scores)
    mean_dir = (sum(dir_scores) / dir_n) if dir_scores else 0.0
    top = max(by_reason, key=by_reason.__getitem__) if by_reason else None
    return {
        "eval_lines": eval_lines,
        "vetoes": sum(by_reason.values()),
        "by_reason": by_reason,
        "top_veto_reason": top,
        "direction_rejected": dir_n,
        "direction_rejected_mean_abs_score": round(mean_dir, 3),
        "real_conviction_discarded": bool(dir_n and mean_dir >= _REAL_CONVICTION),
        "regime": regime,
        "win_rate": round(_resolve_win_rate(win_rate, realized_snapshot), 3),
        "worst_conf_bin_losses": _worst_conf_bin_losses(realized_snapshot),
    }


def _resolve_win_rate(win_rate: float | None, snapshot: dict[str, Any] | None) -> float:
    if win_rate is not None:
        return win_rate
    if not snapshot:
        return 0.5
    wins = snapshot.get("conf_wins", [])
    losses = snapshot.get("conf_losses", [])
    w = int(sum(wins) if wins else 0)
    l = int(sum(losses) if losses else 0)
    total = w + l
    if total:
        return float(w) / total
    return 0.5


def _worst_conf_bin_losses(snapshot: dict[str, Any] | None) -> int:
    if not snapshot:
        return 0
    losses = snapshot.get("conf_losses", [])
    if not isinstance(losses, list):
        return 0
    return max((int(x) for x in losses), default=0)


def build_evidence_prompt(evidence: dict[str, Any]) -> str:
    """Build a chain-of-thought prompt grounding HALIM's self-improvement."""
    return (
        "You are HALIM, the trading architect for Juli (a live trading brain).\n"
        "You just closed a learning cycle. Analyze the EVAL evidence below and "
        "reason step by step (chain of thought) about WHICH gate strangles "
        "entries and whether the cortex had real conviction behind the "
        "rejections. Then recommend bounded parameter adjustments to make Juli "
        "better. Return EXACTLY this JSON, nothing else:\n"
        '{"reasoning": "<your chain of thought: which gate dominates, why the '
        'cortex is right/wrong, what to adjust>", "recommendations": ['
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
    )


def _parse_recs(text: str) -> list[dict[str, Any]]:
    """Parse HALIM's JSON into validated, bounded recommendations only."""
    parsed = _normalize_halim_json(text)
    if not isinstance(parsed, dict):
        return []
    recs = parsed.get("recommendations", [])
    if not isinstance(recs, list):
        return []
    return [
        r for r in recs if isinstance(r, dict) and validate_recommendation(r) is None
    ]


def _http_query(base_url: str, prompt: str) -> dict[str, Any]:
    """POST a prompt to HALIM's /v1/complete and return the parsed JSON."""
    data = json.dumps(
        {"prompt": prompt, "purpose": "evidence_learning", "priority": "high"}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/complete",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    return raw if isinstance(raw, dict) else {}


def fetch_evidence_recs(
    base_url: str,
    evidence: dict[str, Any],
    query: Callable[[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Ask HALIM (CoT) for evidence-grounded bounded recommendations."""
    prompt = build_evidence_prompt(evidence)
    result = (query or _http_query)(base_url, prompt)
    text = str(result.get("text", "")) if isinstance(result, dict) else ""
    recs = _parse_recs(text)
    if recs:
        log.info("HALIM evidence-CoT: %d bounded recs", len(recs))
    return recs


def maybe_fetch_evidence_recs(
    base_url: str, evidence: dict[str, Any], enabled: bool
) -> list[dict[str, Any]]:
    """Gated fetch — no HALIM call, no param write when disabled."""
    if not enabled:
        return []
    try:
        return fetch_evidence_recs(base_url, evidence)
    except Exception as e:
        log.debug("HALIM evidence fetch failed: %s", e)
        return []


__all__ = [
    "EVAL_LOG_PATH",
    "build_evidence_prompt",
    "collect_evidence",
    "fetch_evidence_recs",
    "maybe_fetch_evidence_recs",
    "read_eval_tail",
]
