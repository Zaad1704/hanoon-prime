"""Halim engine — reflex always local; reasoning optional via server checkpoint."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from halim.device import detect_profile, profile_spec
from halim.protocol import MODEL_NAME, PHASES, REFLEX_COMPONENTS, REASONING_COMPONENTS
from halim.active_model import runtime_envelope

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _root() -> Path:
    env = os.getenv("HALIM_REPO_ROOT", "").strip()
    if env:
        return Path(env)
    if (_REPO_ROOT / "models").is_dir():
        return _REPO_ROOT
    return Path.cwd()


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def _asset(path: str) -> Dict[str, Any]:
    p = _root() / path
    return {
        "path": path,
        "exists": p.is_file(),
        "size_kb": round(p.stat().st_size / 1024, 1) if p.is_file() else 0,
    }


def read_phase() -> str:
    ident = _root() / "models/halim_identity.json"
    if ident.is_file():
        try:
            return json.loads(ident.read_text()).get("phase", "newborn")
        except Exception:
            pass
    return "newborn"


def checkpoint_path() -> Optional[Path]:
    raw = os.getenv("HALIM_MODEL_PATH", "halim/data/checkpoints/latest")
    p = Path(raw)
    if not p.is_absolute():
        p = _root() / raw
    if (p / "config.json").is_file() or p.with_suffix(".gguf").is_file():
        return p
    # LoRA-only checkpoint (no merge): config + lora_adapter/
    if (p / "lora_adapter" / "adapter_model.safetensors").is_file():
        return p
    gguf = p if str(p).endswith(".gguf") else None
    if gguf and gguf.is_file():
        return gguf
    return None


def reasoning_available() -> bool:
    spec = profile_spec()
    if spec.get("reflex_only") and not os.getenv("HALIM_FORCE_LM"):
        return False
    return checkpoint_path() is not None


def collect_status() -> Dict[str, Any]:
    """Full Halim engine snapshot — safe on any device."""
    root = _root()
    prof = profile_spec()
    ckpt = checkpoint_path()
    ds = _count_jsonl(
        __import__(
            "core.training_dataset_paths", fromlist=["council_training_dataset_path"]
        ).council_training_dataset_path()
    )

    return {
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": read_phase(),
        "device_profile": prof,
        "dataset_pairs": ds,
        "reflex": {k: _asset({
            "ppo": "models/ppo_trader_replay.zip",
            "proxy": "models/teacher_proxy.joblib",
            "scalper_weights": "models/scalper_weights.json",
        }[k]) for k in REFLEX_COMPONENTS},
        "reasoning": {
            "enabled": reasoning_available(),
            "checkpoint": str(ckpt) if ckpt else None,
            "components": list(REASONING_COMPONENTS),
            "backend": os.getenv("HALIM_LM_BACKEND", "none"),  # mlx | llama_cpp | none
        },
        "architecture": {
            "fast_path": "inline_in_hanoon",
            "slow_path": "halim_server_optional",
            "never_block_trading": True,
            "learn_by_action": True,
        },
        "capabilities": _capability_summary(),
        "runtime_mode": runtime_envelope(),
        "unlock_ladder": _unlock_summary(),
    }


def _unlock_summary() -> Dict[str, Any]:
    try:
        root = _root()
        if str(root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(root))
        from core.halim_unlock import unlock_ladder
        ladder = unlock_ladder()
        return {
            "power_score": ladder.get("power_score"),
            "next_unlock": ladder.get("next_unlock"),
            "modes": {
                k: v.get("mode")
                for k, v in (ladder.get("capabilities") or {}).items()
            },
        }
    except Exception:
        return {}


def _capability_summary() -> Dict[str, Any]:
    """Action counts per capability — safe without importing core."""
    log_path = _root() / "halim/data/actions/action_log.jsonl"
    counts: Dict[str, int] = {}
    if log_path.is_file():
        try:
            with open(log_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json as _json
                        cap = _json.loads(line).get("capability", "?")
                        counts[cap] = counts.get(cap, 0) + 1
                    except Exception:
                        continue
        except Exception:
            pass
    gold_path = _root() / "halim/data/training/action_gold.jsonl"
    gold_n = _count_jsonl(gold_path)
    return {"action_counts": counts, "action_gold_pairs": gold_n}


_CHAT_PURPOSES = frozenset({
    "chat", "commander_chat", "dialogue", "companion", "copilot",
    "decision_text",
})
_NOTIFY_PURPOSES = frozenset({"notify"})
# Postmortem lessons must be SHORT but COMPLETE sentences — give them enough
# tokens to finish under RAM throttle (which can cut to 64-256 tokens).
_POSTMORTEM_PURPOSES = frozenset({"postmortem"})
# Compact structured verdicts — a 512-token default makes these take 60-90s+
# on the 8GB Mac (memory pressure), starving the single inference worker and
# stalling every other Halim call behind them. 72 tokens is plenty for
# {"direction","confidence","reason"}.
_TRADE_PURPOSES = frozenset({"trade_decision", "verdict", "supervision", "evaluation"})
# Purposes whose output MUST be a JSON object (structured decisions/verdicts).
_JSON_PURPOSES = frozenset({
    "entry_decision", "exit_decision", "trade_decision", "verdict",
    "supervision", "evaluation", "model_eval", "winrate_guard",
    "regime",  # Added for market regime classification JSON output
})


def _extract_json(text: str, defaults: Optional[dict] = None) -> Optional[dict]:
    """Extract a valid JSON object from free-form LLM output.

    Handles: surrounding prose, balanced-brace extraction, smart quotes,
    trailing commas, and unquoted keys. Missing required keys are filled
    from `defaults`. Returns None only if no JSON can be recovered.
    """
    import json as _json
    defaults = defaults or {}
    if not text:
        return None
    # Normalize common LLM artifacts
    fixed = text.replace("“", '"').replace("”", '"').replace("'", '"')
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    # Find the first balanced { ... } block
    start = fixed.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(fixed)):
        ch = fixed[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    block = fixed[start:end + 1]
    # Try parse, then light repair (unquoted keys)
    for attempt in (block, re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', block)):
        try:
            obj = _json.loads(attempt)
            if isinstance(obj, dict):
                merged = dict(defaults)
                merged.update({k: v for k, v in obj.items() if v is not None})
                return merged
        except Exception:
            continue
    return None


def complete_reasoning(prompt: str, purpose: str = "reasoning", thinking: bool = False,
                       require_json: Optional[bool] = None,
                       json_defaults: Optional[dict] = None) -> Dict[str, Any]:
    """
    Slow path — Halim LM inference. Returns structured result; never raises.
    When thinking=True, enables Qwen3 chain-of-thought for hard decisions.
    """
    max_tokens = int(os.getenv("HALIM_MAX_TOKENS", "512"))
    temperature = float(os.getenv("HALIM_TEMPERATURE", "0.7"))
    if thinking:
        max_tokens = max(max_tokens, 512)
        temperature = min(temperature, 0.2)
    elif purpose in _CHAT_PURPOSES:
        max_tokens = int(os.getenv("HALIM_CHAT_MAX_TOKENS", "72"))
        temperature = float(os.getenv("HALIM_CHAT_TEMPERATURE", "0.28"))
    elif purpose in _NOTIFY_PURPOSES:
        max_tokens = int(os.getenv("HALIM_NOTIFY_MAX_TOKENS", "120"))
        temperature = float(os.getenv("HALIM_NOTIFY_TEMPERATURE", "0.35"))
    elif purpose in _POSTMORTEM_PURPOSES:
        max_tokens = int(os.getenv("HALIM_POSTMORTEM_MAX_TOKENS", "180"))
        temperature = float(os.getenv("HALIM_POSTMORTEM_TEMPERATURE", "0.3"))
    elif purpose == "playbook":
        # Weekly market playbook — <= 5 rules x <= 12 words + JSON wrapper.
        # A 512-token default made the refresh job time out at 30s.
        max_tokens = int(os.getenv("HALIM_PLAYBOOK_MAX_TOKENS", "128"))
        temperature = float(os.getenv("HALIM_PLAYBOOK_TEMPERATURE", "0.25"))
    elif purpose in _TRADE_PURPOSES:
        max_tokens = int(os.getenv("HALIM_TRADE_MAX_TOKENS", "72"))
        temperature = float(os.getenv("HALIM_TRADE_TEMPERATURE", "0.12"))
    elif purpose in ("entry_decision", "exit_decision"):
        if purpose == "entry_decision":
            max_tokens = int(os.getenv("HALIM_ENTRY_MAX_TOKENS", "72"))
            temperature = float(os.getenv("HALIM_ENTRY_TEMPERATURE", "0.12"))
        else:
            max_tokens = int(os.getenv("HALIM_EXIT_MAX_TOKENS", "72"))
            temperature = float(os.getenv("HALIM_EXIT_TEMPERATURE", "0.12"))
    elif purpose == "regime":
        # Enriched regime prompt needs more room for reasoning about
        # per-ticker detail, dispersion, and volume trends. 48 tokens
        # keeps sub-15s inference while allowing 2-3 sentence reasoning.
        max_tokens = int(os.getenv("HALIM_REGIME_MAX_TOKENS", "48"))
        temperature = float(os.getenv("HALIM_REGIME_TEMPERATURE", "0.2"))
    elif purpose == "winrate_guard":
        # JULI loss-coaching JSON ({"d", "i":[...]}) — 200 tokens is enough;
        # the default 512 got RAM-throttled to 64-256 and truncated the JSON
        # mid-object (no closing brace) → extraction failed → empty advice.
        max_tokens = int(os.getenv("HALIM_WINRATE_MAX_TOKENS", "200"))
        temperature = float(os.getenv("HALIM_WINRATE_TEMPERATURE", "0.2"))
    elif purpose == "system_health":
        # Death-spiral detection advisory — Halim analyzes calibration
        # health, gate state, and recommends thaw/override actions.
        max_tokens = int(os.getenv("HALIM_SYSHEALTH_MAX_TOKENS", "120"))
        temperature = float(os.getenv("HALIM_SYSHEALTH_TEMPERATURE", "0.15"))
    # Default JSON requirement by purpose — structured purposes always
    # return a validated dict, never unparseable text.
    if require_json is None:
        require_json = purpose in _JSON_PURPOSES

    def _postprocess(text: str) -> dict:
        """Attach structured JSON when require_json is set — malformed LLM
        output never reaches the caller as unparseable text."""
        result = {"ok": True, "text": text, "source": "halim_lm",
                  "backend": backend, "purpose": purpose}
        if require_json:
            obj = _extract_json(text, json_defaults)
            if obj is not None:
                result["data"] = obj
            else:
                result["ok"] = False
                result["reason"] = "json_parse_failed"
                result["data"] = json_defaults or {}
        return result

    if not reasoning_available():
        return {
            "ok": False,
            "reason": "no_checkpoint",
            "message": "Halim LM not trained yet — reflex students active; collect dataset",
            "phase": read_phase(),
            "dataset_pairs": collect_status().get("dataset_pairs", 0),
        }

    backend = os.getenv("HALIM_LM_BACKEND", "").lower()
    ckpt = checkpoint_path()
    if backend == "hf" and ckpt:
        try:
            from halim.inference_backend import hf_complete

            text, err = hf_complete(prompt, ckpt, max_tokens=max_tokens, temperature=temperature)
            if text:
                return _postprocess(text)
            return {
                "ok": False,
                "reason": err,
                "message": "HF inference failed — pip install torch transformers",
                "backend": "hf",
            }
        except Exception as exc:
            return {
                "ok": False,
                "reason": "hf_error",
                "message": str(exc)[:200],
                "backend": "hf",
            }

    if backend == "mlx" and ckpt:
        try:
            # Ensure Metal device initialized on this thread
            import mlx.core as _mx
            _mx.set_default_device(_mx.gpu)

            from halim.inference_backend import mlx_complete

            text, err = mlx_complete(prompt, ckpt, max_tokens=max_tokens, temperature=temperature, thinking=thinking)
            if text:
                return _postprocess(text)
            return {
                "ok": False,
                "reason": err,
                "message": "MLX inference failed — check mlx-lm install and checkpoint",
                "backend": "mlx",
            }
        except Exception as exc:
            return {
                "ok": False,
                "reason": "mlx_error",
                "message": str(exc)[:200],
                "backend": "mlx",
            }

    if backend in ("mlx", "llama_cpp"):
        return {
            "ok": False,
            "reason": "backend_not_wired",
            "message": f"Checkpoint found; set HALIM_LM_BACKEND=mlx and install mlx-lm",
            "backend": backend,
        }

    return {
        "ok": False,
        "reason": "backend_not_configured",
        "message": "Set HALIM_LM_BACKEND=mlx when checkpoint is ready",
    }
