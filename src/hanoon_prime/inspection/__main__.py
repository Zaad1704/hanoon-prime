"""Inside Man command line.

Usage:
    python3 -m hanoon_prime.inspection manifest [--json]
    python3 -m hanoon_prime.inspection reanchor [--note TEXT]   (Task 9)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .checks import OK
from .ctx import InspectionContext
from .joints import JOINT_ORDER, run_all


def _resolve_root() -> Path:
    """Repo root: the ancestor holding src/hanoon_prime (editable install)."""
    for p in Path(__file__).resolve().parents:
        if (p / "src" / "hanoon_prime").is_dir():
            return p
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()


def _ctx() -> InspectionContext:
    return InspectionContext(base_dir=ROOT)


def _print_manifest(args: argparse.Namespace) -> int:
    m = run_all(_ctx())
    if args.json:
        payload = asdict(m)
        payload["status"] = m.status
        payload["hard_fails"] = [asdict(r) for r in m.hard_fails]
        payload["anomalies"] = [asdict(r) for r in m.anomalies]
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        out: list[str] = [f"INSIDE-MAN MANIFEST  ts={m.ts:.0f}  head={m.git_head}"]
        out.append(f"status: {m.status}\n")
        for joint in JOINT_ORDER:
            for r in [x for x in m.results if x.joint == joint]:
                out.append(f"  [{r.status:<11}] {r.joint}.{r.name}  {r.detail}")
        out.append("")
        out.append(
            "HARD FAILS: "
            + str([f"{r.joint}.{r.name}" for r in m.hard_fails] or "none")
        )
        out.append(
            "ANOMALIES : " + str([f"{r.joint}.{r.name}" for r in m.anomalies] or "none")
        )
        sys.stdout.write("\n".join(out) + "\n")
    if m.status == OK:
        return 0
    if m.hard_fails:
        return 2
    if m.anomalies:
        return 4
    return 3


def _print_heal(args: argparse.Namespace) -> int:
    from .heal import heal as heal_actions

    manifest = run_all(_ctx())
    for line in heal_actions(_ctx(), manifest, dry_run=args.dry_run):
        sys.stdout.write(line + "\n")
    return 0


def _print_reanchor(args: argparse.Namespace) -> int:
    from .reanchor import reanchor_if_broken

    _, outcome = reanchor_if_broken(_ctx())
    sys.stdout.write(outcome + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch to the requested inspection subcommand."""
    ap = argparse.ArgumentParser(description="Inside Man facility inspection")
    sub = ap.add_subparsers(dest="command", required=True)
    pm = sub.add_parser("manifest", help="print the full check manifest")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=_print_manifest)
    ph = sub.add_parser("heal", help="gated auto-heal of mechanical services")
    ph.add_argument(
        "--dry-run", action="store_true", help="show candidate actions, take none"
    )
    ph.set_defaults(func=_print_heal)
    pr = sub.add_parser(
        "reanchor", help="re-anchor journal chain if broken (boot step)"
    )
    pr.set_defaults(func=_print_reanchor)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
