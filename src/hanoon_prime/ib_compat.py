"""hanoon_prime.ib_compat — ib_insync import compatibility shim.

On Python 3.14+, the eventkit library (dependency of ib_insync) calls
asyncio.get_event_loop() at import time, which raises RuntimeError
when no event loop is running in the main thread.

This module pre-creates an event loop and sets it as the current
thread's event loop before importing ib_insync.

If ib_insync is not installed, ``ib`` is None and ``_ib_available`` is
False.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

_ib_available: bool = False
ib: Any = None

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

# Python 3.14 deprecated asyncio.get_event_loop_policy(); ib_insync's eventkit
# dependency calls it at import time. Under the project's strict
# `filterwarnings = error::DeprecationWarning` (pytest.ini) that raises,
# which breaks ib_insync import and every downstream
# `from ib_insync import …` site (netting-guard order placement, scanner
# subscriptions, etc.). Downgrade _this one_ third-party deprecation to a
# normal, non-fatal warning so the import succeeds and ib_insync is cached in
# sys.modules for all later `from ib_insync import …` calls.
warnings.filterwarnings(
    "default",
    message=r".*get_event_loop_policy.*",
    category=DeprecationWarning,
)

try:
    import ib_insync as ib

    _ib_available = True
except Exception:
    ib = None
