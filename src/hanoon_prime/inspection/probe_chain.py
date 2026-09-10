"""probe_chain — journal reader + hash-chain anchor scan.

Split from probe.py to respect the 200-line src-file rule. probe.py
re-exports journal / journal_tail / journal_chain_state.
"""

from __future__ import annotations

from typing import Any

from ..memory import Journal
from .ctx import InspectionContext


def journal(ctx: InspectionContext) -> Journal:
    """Append-only journal reader (sized to the real file)."""
    if "journal" not in ctx.memo:
        ctx.memo["journal"] = Journal(ctx.journal_path)
    value = ctx.memo["journal"]
    if isinstance(value, Journal):
        return value
    j = Journal(ctx.journal_path)
    ctx.memo["journal"] = j
    return j


def journal_tail(ctx: InspectionContext, n: int) -> list[dict[str, Any]]:
    """Last n journal rows as dicts."""
    return journal(ctx).tail(n)


def journal_chain_state(ctx: InspectionContext) -> dict[str, Any]:
    """Hash-chain anchor + contiguity after the last re-anchor / break."""
    key = "chain"
    if key in ctx.memo:
        value = ctx.memo[key]
        return dict(value) if isinstance(value, dict) else {}
    entries = journal(ctx).entries()
    last_break = -1
    last_reseed = -1
    breaks = 0
    prev: str | None = None
    for i, e in enumerate(entries):
        if e.get("event") == "chain_reseed":
            last_reseed = i
        if e.get("prev_hash") != prev:
            breaks += 1
            last_break = i
        prev = e.get("hash")
    anchor = max(
        last_reseed + 1 if last_reseed >= 0 else 0,
        last_break + 1 if last_break >= 0 else 0,
    )
    gaps = 0
    p: str | None = entries[anchor - 1].get("hash") if anchor > 0 else None
    for e in entries[anchor:]:
        if e.get("prev_hash") != p:
            gaps += 1
        p = e.get("hash")
    ctx.memo[key] = {
        "entries": len(entries),
        "breaks": breaks,
        "last_break_seq": entries[last_break].get("seq") if last_break >= 0 else None,
        "last_reseed_seq": (
            entries[last_reseed].get("seq") if last_reseed >= 0 else None
        ),
        "anchor_seq": entries[anchor].get("seq") if anchor < len(entries) else None,
        "gaps_after_anchor": gaps,
    }
    value = ctx.memo[key]
    return dict(value) if isinstance(value, dict) else {}
