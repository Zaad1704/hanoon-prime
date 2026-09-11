"""tests/test_anti_monkeypatch.py — anti-monkeypatch contract (JULI 2.0).

Three layers:

1. Critical-singleton denylist: tests may not monkeypatch the risk/stdp
   critical singletons unless explicitly allowed with
   ``# allow: monkeypatch <symbol>`` on the same or preceding line.
2. Contract no-patch rule: test_contract.py must never monkeypatch
   hanoon_prime.inspection.* or the module under test.
3. Production self-patching guard: no src module rewrites its own
   import-time bindings at module scope.

The scanner lives in ``scripts/check_monkeypatch.py``; these tests bind it
so CI runs it on every commit and the contract stays enforced locally too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_monkeypatch.py"


def _load_scanner():
    spec = importlib.util.spec_from_file_location("check_monkeypatch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_critical_singleton_denylist_enforced():
    mod = _load_scanner()
    assert mod.main(["--src-only"]) == 0, "src self-patching guard must pass"
    assert mod.main([]) == 0, (
        "anti-monkeypatch contract violated — review scripts/check_monkeypatch.py "
        "output for the offending patch sites"
    )


def test_scanner_detects_unauthorized_patch():
    """The denylist must flag an un-annotated denylist patch."""

    class _FakePath:
        name = "test_dummy.py"
        text = (
            "def test_x(self, monkeypatch):\n"
            '    monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "both")\n'
        )

        def read_text(self):
            return self.text

    mod = _load_scanner()
    assert mod._check_test_file(_FakePath()), "denylist patch must be flagged"


def test_scanner_allows_annotated_patch():
    class _FakePath:
        name = "test_dummy.py"
        text = (
            "def test_x(self, monkeypatch):\n"
            '    monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "both")'
            "  # allow: monkeypatch TRADING_CONFIG\n"
        )

        def read_text(self):
            return self.text

    mod = _load_scanner()
    assert mod._check_test_file(_FakePath()) == [], "annotated patch must pass"


def test_scanner_flags_contract_no_patch():
    class _FakePath:
        name = "test_contract.py"
        text = (
            "def test_x(self, monkeypatch):\n"
            '    monkeypatch.setattr(hanoon_prime.inspection.system.run, "x")\n'
        )

        def read_text(self):
            return self.text

    mod = _load_scanner()
    assert mod._check_contract_no_patch(
        _FakePath()
    ), "contract test patching inspection must be flagged"


def test_scanner_flags_self_patch():
    class _FakePath:
        name = "bad_module.py"
        text = "sys.modules[__name__].foo = lambda: 1\n"

        def read_text(self):
            return self.text

    mod = _load_scanner()
    assert mod._check_src_no_self_patch(
        _FakePath()
    ), "import-time self-patch must be flagged"
