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
from typing import Any

_ib_available: bool = False
ib: Any = None

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

try:
    import ib_insync as ib

    _ib_available = True
except Exception:
    ib = None
