"""Probe the live Gateway for the authoritative scan-code whitelist.

Read-only: connects with an unused clientId, calls reqScannerParameters,
prints the STK / STK.US.MAJOR scan codes, then disconnects. No orders,
no subscriptions that persist.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from hanoon_prime.ib_compat import ib  # noqa: E402

if ib is None:
    print("ib_insync not installed")
    raise SystemExit(1)


def main() -> int:
    client = ib.IB()
    client.connect("127.0.0.1", 4002, clientId=97)
    if not client.isConnected():
        print("connect failed")
        return 1
    try:
        client.reqScannerParameters()
        xml = client.reqScannerParams  # delivered via wrapper._endReq
        if not xml:
            print("no scanner params received")
            return 1
        data = json.loads(xml)
        scan_codes = data.get("ScanCodeList", {}).get("InstrumentScanCode", [])
        want = {"STK", "STOCK.EU"}
        for inst in scan_codes:
            if inst.get("instrument") not in want:
                continue
            for loc in inst.get("Location", []):
                if loc.get("locationCode") not in {"STK.US.MAJOR", "STK.US"}:
                    continue
                codes = [e["scanCode"] for e in loc.get("ScanCode", [])]
                print(f"locationCode={loc.get('locationCode')}")
                for code in sorted(codes):
                    print(f"  {code}")
        return 0
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
