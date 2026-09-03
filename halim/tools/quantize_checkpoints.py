#!/usr/bin/env python3
"""
halim/tools/quantize_checkpoints.py — Generate 3-bit and 2-bit MLX checkpoints.

The PrecisionManager can drop from 4-bit to 3-bit/2-bit under memory pressure,
but the quantized checkpoint files must ALREADY exist on disk. This script
generates them by loading the deployed LoRA adapter, merging it into the base
model, and saving at the target precision.

Usage:
    python3 halim/tools/quantize_checkpoints.py --bits 3
    python3 halim/tools/quantize_checkpoints.py --bits 2
    python3 halim/tools/quantize_checkpoints.py --all          # Generate both

Output:
    halim/data/checkpoints/toddler_v3_kaggle_3bit/
    halim/data/checkpoints/toddler_v3_kaggle_2bit/

Requires ~3GB free RAM during conversion. Run during low-activity periods
(non-RTH) to avoid competing with trading.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = ROOT / "halim" / "data" / "checkpoints"
LATEST_SYMLINK = CKPT_DIR / "latest"


def _load_config(checkpoint: Path) -> Dict[str, Any]:
    cfg_file = checkpoint / "config.json"
    if cfg_file.exists():
        return json.loads(cfg_file.read_text())
    return {}


def _find_adapter(checkpoint: Path) -> Optional[Path]:
    """Find the LoRA adapter safetensors file."""
    cfg = _load_config(checkpoint)
    adapter_rel = cfg.get("adapter_path", "")
    if adapter_rel:
        adapter_path = checkpoint / adapter_rel
    else:
        adapter_path = checkpoint
    adapter_file = adapter_path / "adapters.safetensors"
    if adapter_file.exists():
        return adapter_file
    return None


def quantize_checkpoint(bits: int, checkpoint: Path, dry_run: bool = False) -> bool:
    """Load the deployed model + LoRA adapter and save at target precision."""
    config = _load_config(checkpoint)
    base_model = config.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    out_name = f"{checkpoint.name}_{bits}bit"
    out_dir = CKPT_DIR / out_name

    if out_dir.exists() and not dry_run:
        sz = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / (1024**3)
        print(f"  ✅ {out_name} already exists ({sz:.1f}GB). Skip (use --force to regenerate).")
        return True

    print(f"\n{'='*60}")
    print(f"  Quantizing {checkpoint.name} → {bits}-bit")
    print(f"  Base model: {base_model}")
    print(f"  Output: {out_name}")
    print(f"{'='*60}\n")

    if dry_run:
        print("  DRY RUN — would generate:")
        print(f"    {out_dir}/")
        return True

    start = time.time()

    # Step 1: Load the LoRA adapter
    adapter_file = _find_adapter(checkpoint)
    if adapter_file:
        print(f"  📦 Found LoRA adapter: {adapter_file}")
    else:
        print(f"  ⚠️  No LoRA adapter found — quantizing base model only")

    try:
        import mlx.core as mx
        from mlx_lm import load

        # Step 2: Load model normally (4-bit), then quantize post-load
        print(f"  🔄 Loading {base_model}...")
        sys.stdout.flush()
        model, tokenizer = load(
            base_model,
            adapter_path=str(adapter_file.parent) if adapter_file else None,
        )

        # Step 3: Quantize to target bits using mlx.core
        print(f"  🔧 Quantizing to {bits}-bit...")
        sys.stdout.flush()
        quantized_weights = {}
        for name, param in model.parameters() if hasattr(model, 'parameters') else model.trainable_weights().items():
            if isinstance(param, mx.array):
                group_size = 64
                bits_val = bits
                quantized_weights[name] = mx.quantize(param, group_size=group_size, bits=bits_val)
            else:
                quantized_weights[name] = param

        # Step 4: Save quantized model
        print(f"  💾 Saving to {out_dir}...")
        sys.stdout.flush()
        out_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(out_dir / "model.safetensors"), quantized_weights)

        # Save tokenizer
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(str(out_dir))
        else:
            import json as _json
            (out_dir / "tokenizer_config.json").write_text(_json.dumps({}))

        # Save config metadata
        config_path = out_dir / "config.json"
        cfg = {"precision": f"{bits}-bit", "quantized_from": str(checkpoint),
               "quantized_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
               "base_model": config.get("base_model", base_model)}
        config_path.write_text(json.dumps(cfg, indent=2))

        elapsed = time.time() - start
        model_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / (1024**3)
        print(f"  ✅ Done in {elapsed:.0f}s — {model_size:.2f}GB")
        return True

    except Exception as exc:
        print(f"  ❌ Quantization failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate lower-precision Halim checkpoints")
    parser.add_argument("--bits", type=int, choices=[2, 3], help="Target precision (2 or 3)")
    parser.add_argument("--all", action="store_true", help="Generate both 2-bit and 3-bit")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint name (default: latest symlink target)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")
    args = parser.parse_args()

    if not args.bits and not args.all:
        parser.error("Specify --bits 2, --bits 3, or --all")

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint = CKPT_DIR / args.checkpoint
    else:
        checkpoint = LATEST_SYMLINK.resolve()
    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        sys.exit(1)
    print(f"📁 Using checkpoint: {checkpoint}")

    bits_list = [2, 3] if args.all else [args.bits]
    success = True

    for bits in bits_list:
        out_name = f"{checkpoint.name}_{bits}bit"
        out_dir = CKPT_DIR / out_name
        if out_dir.exists() and not args.force:
            sz = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / (1024**3)
            print(f"  ✅ {out_name} already exists ({sz:.1f}GB). Use --force to regenerate.")
            continue
        ok = quantize_checkpoint(bits, checkpoint, dry_run=args.dry_run)
        if not ok:
            success = False

    if success:
        print(f"\n✅ All requested quantizations complete.")
        print(f"   The PrecisionManager will now be able to downgrade to lower precision")
        print(f"   when memory pressure reaches ORANGE/RED levels.")
    else:
        print(f"\n⚠️  Some quantizations failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
