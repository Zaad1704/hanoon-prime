"""Re-anchor the journal chain after an authorized stop (start.command step)."""

from __future__ import annotations

import time

from ..memory import Journal
from .ctx import InspectionContext


def reanchor_if_broken(ctx: InspectionContext) -> tuple[bool, str]:
    """Append a chain_reseed anchor iff the journal hash chain is broken."""
    j = Journal(ctx.journal_path)
    try:
        intact = j.verify_chain()
    except Exception as exc:  # never raise through the boot step
        return False, f"verify failed: {exc}"
    if not intact:
        j.append({"event": "chain_reseed", "ts": time.time(), "reason": "hash_break"})
        return True, "chain re-anchored (hash_break)"
    return False, "chain intact"
