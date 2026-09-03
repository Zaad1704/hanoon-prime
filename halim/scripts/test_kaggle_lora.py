#!/usr/bin/env python3
"""Direct MLX inference test with Halim Kaggle LoRA."""

from pathlib import Path
import json
import sys

ROOT = Path("/Users/mdsabersajib/Downloads/tradingbot")
sys.path.insert(0, str(ROOT / "halim"))

from halim.engine import checkpoint_path, complete_reasoning, collect_status

ckpt = checkpoint_path()
print(f"Checkpoint: {ckpt}")
print(f"Backend: mlx")

# Try a simple prompt
prompts = [
    {
        "id": "simple_greeting",
        "purpose": "chat",
        "prompt": "Say hello and introduce yourself as Halim."
    },
    {
        "id": "entry_test",
        "purpose": "entry_decision",
        "prompt": (
            "TASK: entry_decision\n"
            "spike=2.3x scan=76 profit_prob=0.83 fakeout=0.20 ticker=PLUG sector=Energy conviction=88%\n"
            "Source: commander IB lessons"
        ),
    },
]

for p in prompts:
    print(f"\n{'='*60}")
    print(f"Probe: {p['id']} (purpose={p['purpose']})")
    print(f"Prompt: {p['prompt'][:80]}...")
    print(f"{'='*60}")
    
    result = complete_reasoning(p["prompt"], purpose=p["purpose"])
    
    if result.get("ok"):
        print(f"✓ OK")
        print(f"Output: {result.get('text', '')[:200]}")
    else:
        print(f"✗ FAILED")
        print(f"Reason: {result.get('reason', 'unknown')}")
        print(f"Message: {result.get('message', '')[:200]}")

print(f"\n{'='*60}")
print("Engine status:")
status = collect_status()
print(f"  Reasoning enabled: {status['reasoning']['enabled']}")
print(f"  Checkpoint: {status['reasoning']['checkpoint']}")
print(f"  Backend: {status['reasoning']['backend']}")
print(f"  Phase: {status['phase']}")
print(f"  Device: {status['device_profile']['profile']}")
