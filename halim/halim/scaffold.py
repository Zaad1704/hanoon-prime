"""
Halim scaffold — HuggingFace/MLX registry IDs.

Two sets of constants:
  1. SCAFFOLD_*    — Open-weight bases for *training* new LoRA adapters (0.5B).
  2. DEPLOYED_*   — The model identifier actually running in production
                     (Qwen3.5-4B MoE, 4-bit MLX, ~1B active params per token).

M. A. Halim is the owned product (see halim.protocol.MODEL_NAME).
Never use scaffold names in user-facing logs or UI.
"""

from __future__ import annotations

# ── Training scaffold (0.5B) ──────────────────────────────────────────
# Used by train_toddler_colab.py and prepare_sft.py for LoRA training.
# These are the CHEAP / FAST open-weight bases, NOT what runs in production.
SCAFFOLD_HF = "Qwen/Qwen2.5-0.5B-Instruct"
SCAFFOLD_MLX_4BIT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# ── Deployed model: Qwen3.5-4B MoE (4-bit MLX) ────────────────────────
# What actually runs on the Mac during live/replay trading.
# DEPLOYED_MODEL_MLX_4BIT is the LOCAL checkpoint path (base + LoRA in one
# dir — halim/data/checkpoints/qwen3.5_4b_v2, symlinked as "latest"). The
# HF id "mlx-community/Qwen3.5-4B-4bit-mlx" is NOT public (401) and the old
# Qwen3-1.7B base caused dimension mismatch + memory ballooning. MoE
# activates only ~1B params per token → RSS stays ~500-650MB.
# DO NOT change these unless you switch the deployed checkpoint.
DEPLOYED_MODEL_HF = "Qwen/Qwen3.5-4B"
DEPLOYED_MODEL_MLX_4BIT = "halim/data/checkpoints/qwen3.5_4b_v2"
