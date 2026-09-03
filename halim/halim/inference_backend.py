"""Optional Halim LM backends — MLX first, lazy-loaded in halim serve."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from halim.scaffold import DEPLOYED_MODEL_HF, DEPLOYED_MODEL_MLX_4BIT, SCAFFOLD_HF, SCAFFOLD_MLX_4BIT

_model_cache: Dict[str, Any] = {}
# MLX/HF generate is not thread-safe — parallel /v1/complete during RTH segfaults serve.
_inference_lock = threading.Lock()


def _load_manifest(checkpoint: Path) -> Dict[str, Any]:
    cfg = checkpoint / "config.json"
    if cfg.is_file():
        try:
            return json.loads(cfg.read_text())
        except Exception:
            pass
    return {}


def _resolve_paths(checkpoint: Path) -> Tuple[str, Optional[str]]:
    manifest = _load_manifest(checkpoint)
    base = manifest.get("base_model") or os.getenv("HALIM_BASE_MODEL", SCAFFOLD_MLX_4BIT)
    # SAFEGUARD: On MLX backend, never load full-precision model on 8GB Mac.
    # If someone accidentally configures full-precision, force 4-bit.
    if os.getenv("HALIM_LM_BACKEND", "").lower() == "mlx":
        _full_precision = {
            "Qwen/Qwen3.5-4B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        }
        if base in _full_precision:
            import logging
            logging.warning(f"[Halim] Forcing 4-bit: '{base}' → '{DEPLOYED_MODEL_MLX_4BIT}' (8GB RAM safe)")
            base = DEPLOYED_MODEL_MLX_4BIT
    _validate_base_model(base, checkpoint)
    adapter = manifest.get("adapter_path")
    if adapter:
        ap = checkpoint / adapter if not Path(adapter).is_absolute() else Path(adapter)
        if ap.is_dir() and (ap / "adapters.safetensors").is_file():
            return str(base), str(ap)
    if (checkpoint / "adapters.safetensors").is_file():
        return str(base), str(checkpoint)
    return str(base), None


def _prefer_adapter_inference() -> bool:
    return os.getenv("HALIM_SERVE_PREFER_ADAPTER", "true").lower() in ("1", "true", "yes")


def mlx_complete(
    prompt: str,
    checkpoint: Path,
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    thinking: bool = False,
) -> Tuple[Optional[str], str]:
    """Generate text with mlx-lm. Returns (text, error_reason)."""
    # Thinking mode needs more tokens for chain-of-thought
    if thinking:
        max_tokens = max(max_tokens, 512)
    # Initialize Metal device for this thread (required before mlx_lm operations)
    try:
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
    except Exception:
        pass

    try:
        from mlx_lm import generate, load
    except ImportError:
        return None, "mlx_lm_not_installed"

    # ── RAM-aware max_tokens throttle ──────────────────────────────
    # When system memory is under pressure, reduce output length to
    # prevent KV cache blowup and swap thrashing.
    try:
        import psutil as _psutil
        _vm = _psutil.virtual_memory()
        _avail_mb = _vm.available / (1024 * 1024)
        if _avail_mb < 400:
            max_tokens = min(max_tokens, 64)
        elif _avail_mb < 800:
            max_tokens = min(max_tokens, 128)
        elif _avail_mb < 1500:
            max_tokens = min(max_tokens, 256)
    except Exception:
        pass

    merged = None if _prefer_adapter_inference() else _merged_model_dir(checkpoint)
    adapter_only = _adapter_dir(checkpoint) if _prefer_adapter_inference() else None
    if merged is None and adapter_only is None:
        merged = _merged_model_dir(checkpoint)
    key = str((merged or adapter_only or checkpoint).resolve())
    with _inference_lock:
        if key not in _model_cache:
            try:
                if merged:
                    model, tokenizer = load(str(merged))
                else:
                    base, adapter = _resolve_paths(checkpoint)
                    if adapter_only:
                        adapter = str(adapter_only)
                    if adapter:
                        model, tokenizer = load(base, adapter_path=adapter)
                    else:
                        model, tokenizer = load(base)
                _model_cache[key] = (model, tokenizer)
            except Exception as exc:
                # Colab PEFT 0.19+ adapters may not load on mlx-lm — fall back to merged weights.
                if merged is None:
                    merged = _merged_model_dir(checkpoint)
                if merged is not None:
                    try:
                        mkey = str(merged.resolve())
                        if mkey not in _model_cache:
                            model, tokenizer = load(str(merged))
                            _model_cache[mkey] = (model, tokenizer)
                        else:
                            model, tokenizer = _model_cache[mkey]
                        _model_cache[key] = (model, tokenizer)
                    except Exception as exc2:
                        return None, f"load_failed:{exc2}"[:120]
                else:
                    return None, f"load_failed:{exc}"[:120]

        model, tokenizer = _model_cache[key]
        try:
            from mlx_lm.sample_utils import make_sampler

            # Apply chat template — instruct models need proper role markers
            # (Qwen: <|im_start|>, Llama: <|start_header_id|>, etc.)
            # Qwen3 supports enable_thinking=True for chain-of-thought reasoning
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except TypeError:
                # Older tokenizers don't support enable_thinking
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            # Wrap generate in a thread with timeout so memory pressure
            # (swap > 30s) doesn't hang the serve worker for 90+ seconds.
            _INFERENCE_TIMEOUT = float(os.getenv("MLX_GENERATE_TIMEOUT_SEC", "30.0"))

            # Run MLX generate directly on this thread.
            # Metal stream context is NOT inherited by child threads, so threading
            # is avoided (the timeout thread wrapper was removed for this reason).
            import mlx.core as _mx
            _mx.set_default_device(_mx.gpu)

            result = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
                verbose=False,
            )
            text = (result or "").strip()
            return text or None, "ok"
        except Exception as exc:
            return None, f"generate_failed:{exc}"[:120]
        finally:
            # ── KV cache eviction — ONLY under real memory pressure ────
            # Qwen3.5-4B is a MoE (user directive): only ~1B params activate
            # per token, RSS stays ~500-800MB. Clearing the GPU cache after
            # EVERY inference evicts the resident model → the next call
            # re-faults 2.8GB from disk → 60-90s → timeouts → Halim appears
            # DOWN. Clear only when the system is genuinely memory-starved.
            try:
                import psutil as _pu
                _vm = _pu.virtual_memory()
                _clear_cache = _vm.available < 600 * 1024 * 1024  # <600MB free
            except Exception:
                _clear_cache = False
            if _clear_cache:
                try:
                    import mlx.core as _mx
                    _mx.clear_cache()
                except Exception:
                    pass
                try:
                    import gc as _gc
                    _gc.collect()
                except Exception:
                    pass


def _adapter_dir(checkpoint: Path) -> Optional[Path]:
    manifest = _load_manifest(checkpoint)
    rel = manifest.get("adapter_path", "lora_adapter")
    for candidate in (checkpoint / rel, checkpoint / "lora_adapter"):
        if candidate.is_dir() and (
            (candidate / "adapter_model.safetensors").is_file()
            or (candidate / "adapters.safetensors").is_file()
        ):
            return candidate
    # Adapter files directly in checkpoint root (no subdirectory)
    if (checkpoint / "adapter_model.safetensors").is_file():
        return checkpoint
    return None


def _merged_model_dir(checkpoint: Path) -> Optional[Path]:
    manifest = _load_manifest(checkpoint)
    rel = manifest.get("merged_path", "merged")
    merged = checkpoint / rel
    if merged.is_dir() and (merged / "config.json").is_file():
        return merged
    if (checkpoint / "config.json").is_file() and (checkpoint / "model.safetensors.index.json").is_file():
        return checkpoint
    if (checkpoint / "config.json").is_file() and list(checkpoint.glob("*.safetensors")):
        return checkpoint
    return None


def hf_complete(
    prompt: str,
    checkpoint: Path,
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Tuple[Optional[str], str]:
    """Generate with HuggingFace merged model or LoRA adapter (Colab export)."""
    # ── RAM-aware max_tokens throttle ──────────────────────────────
    try:
        import psutil as _psutil
        _vm = _psutil.virtual_memory()
        _avail_mb = _vm.available / (1024 * 1024)
        if _avail_mb < 400:
            max_tokens = min(max_tokens, 64)
        elif _avail_mb < 800:
            max_tokens = min(max_tokens, 128)
        elif _avail_mb < 1500:
            max_tokens = min(max_tokens, 256)
    except Exception:
        pass

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None, "transformers_not_installed"

    manifest = _load_manifest(checkpoint)
    model_dir = _merged_model_dir(checkpoint)
    adapter_dir = _adapter_dir(checkpoint) if not model_dir else None
    if not model_dir and not adapter_dir:
        return None, "no_merged_or_adapter_in_checkpoint"

    base_model = manifest.get("base_model") or os.getenv("HALIM_BASE_MODEL", SCAFFOLD_HF)
    tokenizer_source = base_model
    if model_dir and (model_dir / "tokenizer.json").is_file():
        tokenizer_source = str(model_dir)

    key = f"{checkpoint.resolve()}|{model_dir}|{adapter_dir}|{tokenizer_source}"
    with _inference_lock:
        if key not in _model_cache:
            try:
                if torch.backends.mps.is_available():
                    device = "mps"
                    dtype = torch.float16
                elif torch.cuda.is_available():
                    device = "cuda"
                    dtype = torch.float16
                else:
                    device = "cpu"
                    dtype = torch.float32
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if model_dir:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_dir), torch_dtype=dtype, trust_remote_code=True,
                    ).to(device)
                else:
                    from peft import PeftModel
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model, torch_dtype=dtype, trust_remote_code=True,
                    ).to(device)
                    model = PeftModel.from_pretrained(model, str(adapter_dir))
                model.eval()
                _model_cache[key] = (model, tokenizer, device)
            except Exception as exc:
                return None, f"load_failed:{exc}"[:120]

        model, tokenizer, device = _model_cache[key]
        try:
            messages = [{"role": "user", "content": prompt}]
            text_in = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text_in, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return text or None, "ok"
        except Exception as exc:
            return None, f"generate_failed:{exc}"[:120]
        finally:
            # ── Clear MPS/GPU cache after every inference ─────────────
            # Prevents KV cache from growing unbounded during RTH trading.
            try:
                import torch as _torch
                if _torch.backends.mps.is_available():
                    _torch.mps.empty_cache()
            except Exception:
                pass
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass


def _validate_base_model(base: str, checkpoint: Path) -> None:
    """Warn at most once if resolved base model doesn't match the deployed model."""
    ckpt_name = checkpoint.name
    # tolerant check: "1.7b" or "1.5b" in base → matches deployed
    base_tag = base.lower()
    deployed_tags = ("1.7", "1.5")
    scaffold_tags = ("0.5",)
    manifest = _load_manifest(checkpoint)
    manifest_base = manifest.get("base_model", "")

    # Check if manifest expects a larger model but resolved to a smaller one
    is_manifest_large = any(t in manifest_base.lower() for t in ("1.7", "1.5"))
    is_resolved_small = any(t in base_tag for t in scaffold_tags)
    if is_manifest_large and is_resolved_small:
        import logging
        logging.warning(
            f"[Halim] ⚠️ CHECKPOINT '{ckpt_name}' expects {manifest_base} base "
            f"but resolved to {base}. "
            f"This would load larger LoRA adapters on a smaller scaffold — producing GARBAGE. "
            f"Fix: ensure HALIM_MODEL_PATH points to a valid checkpoint."
        )
