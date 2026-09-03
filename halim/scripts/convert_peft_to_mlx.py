#!/usr/bin/env python3
"""Convert PEFT LoRA (adapter_model.safetensors) → MLX format (adapters.safetensors)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "halim/data/checkpoints"


def convert_peft_to_mlx(checkpoint_name: str) -> dict:
    ckpt = CKPT_DIR / checkpoint_name
    lora_dir = ckpt / "lora_adapter"

    peft_path = lora_dir / "adapter_model.safetensors"
    peft_cfg_path = lora_dir / "adapter_config.json"
    if not peft_path.is_file():
        raise FileNotFoundError(f"No PEFT adapter at {peft_path}")

    from safetensors.torch import load_file, save_file

    # ── Read PEFT config to compute correct MLX scale ──────────────────
    # PEFT applies scaling = lora_alpha / rank at runtime.
    # MLX stores raw scale in adapter_config.json. If we copy lora_alpha
    # directly (e.g. 16.0 instead of 16/8 = 2.0) the LoRA delta is 8x too
    # large and corrupts the merged model.
    peft_cfg = json.loads(peft_cfg_path.read_text()) if peft_cfg_path.is_file() else {}
    lora_r = peft_cfg.get("r", 8)
    lora_alpha = peft_cfg.get("lora_alpha", 16)
    correct_scale = lora_alpha / lora_r  # e.g. 16/8 = 2.0
    print(f"PEFT r={lora_r} alpha={lora_alpha} → MLX scale={correct_scale}")

    peft_weights = load_file(str(peft_path))
    peft_keys = list(peft_weights.keys())
    print(f"PEFT adapter: {len(peft_keys)} keys")

    mlx_weights = {}
    for key in peft_keys:
        weight = peft_weights[key]
        # PEFT: base_model.model.model.layers.N.self_attn.q_proj.lora_X.weight
        # MLX:  model.layers.N.self_attn.q_proj.lora_a
        mlx_key = key.replace("base_model.model.", "").replace(".weight", "")
        if mlx_key.endswith("lora_A"):
            mlx_key = mlx_key.replace("lora_A", "lora_a")
            weight = weight.T.contiguous()  # (r, in_features) -> (in_features, r)
        elif mlx_key.endswith("lora_B"):
            mlx_key = mlx_key.replace("lora_B", "lora_b")
            weight = weight.T.contiguous()  # (out_features, r) -> (r, out_features)
        mlx_weights[mlx_key] = weight.to(torch.float16).contiguous()

    print(f"MLX adapter: {len(mlx_weights)} keys")
    for k, v in sorted(mlx_weights.items())[:5]:
        print(f"  {k}: {v.shape} {v.dtype}")

    mlx_path = lora_dir / "adapters.safetensors"
    save_file(mlx_weights, str(mlx_path))
    mb = mlx_path.stat().st_size / 1e6
    print(f"\nSaved: {mlx_path} ({mb:.1f} MB)")

    # ── Write MLX adapter_config.json with correct scale ───────────────
    mlx_cfg = {
        "fine_tune_type": "lora",
        "model": peft_cfg.get("base_model_name_or_path", "unknown"),
        "num_layers": "auto",  # mlx-lm reads this from the model if "auto"
        "lora_parameters": {
            "rank": lora_r,
            "dropout": peft_cfg.get("lora_dropout", 0.0),
            "scale": correct_scale,  # ← CRITICAL: alpha/r, not raw alpha
        },
        "seed": 0,
        "train": False,
    }
    (lora_dir / "adapter_config.json").write_text(json.dumps(mlx_cfg, indent=2))
    print(f"MLX adapter_config.json written with scale={correct_scale}")

    mlx_layers = sorted(set(k.split(".")[2] for k in mlx_weights if k.startswith("model.layers.")))
    print(f"Layers: {mlx_layers[0]}-{mlx_layers[-1]} ({len(mlx_layers)} total)")

    return {
        "ok": True,
        "peft_keys": len(peft_keys),
        "mlx_keys": len(mlx_weights),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "mlx_scale": correct_scale,
        "output": str(mlx_path),
    }


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "toddler_v3_kaggle"
    result = convert_peft_to_mlx(name)
    print(json.dumps(result, indent=2))
