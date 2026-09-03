#!/usr/bin/env python3
"""One-time download + cache for Qwen3.5-4B-MLX-4bit-MoE."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from halim.scaffold import DEPLOYED_MODEL_MLX_4BIT
print(f"Downloading {DEPLOYED_MODEL_MLX_4BIT}...")
try:
    from mlx_lm import load
    model, tokenizer = load(DEPLOYED_MODEL_MLX_4BIT)
    print(f"✅ 4-bit model loaded ({type(model).__name__})")
    del model, tokenizer
    print("✅ Model cached — ready for Halim")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)
