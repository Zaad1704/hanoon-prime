#!/usr/bin/env python3
"""Halim serve — active runtime server (status + reasoning + learn/write). Not inference-only."""

from __future__ import annotations

import sys
from pathlib import Path

_pkg = Path(__file__).resolve().parents[1]
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Process, Queue as MPQueue, cpu_count
from queue import PriorityQueue, Empty
from typing import Any, Dict, List

from halim.active_model import enforce_active_runtime, runtime_envelope
from halim.engine import collect_status, complete_reasoning
from halim.protocol import DEFAULT_HOST, DEFAULT_PORT, MODEL_NAME, PROTOCOL_VERSION

# ── Inference Priority Levels ─────────────────────────────────────────
# CRITICAL = entry qualification during active trading (must be served FIRST)
# HIGH    = regime classification, winrate diagnosis
# MEDIUM   = thinking supervision, exit consultation
# LOW     = research daemon, chat, code generation
PRIORITY_CRITICAL = 0
PRIORITY_HIGH     = 1
PRIORITY_MEDIUM   = 2
PRIORITY_LOW      = 3

# Multi-process worker pool threshold. 4000MB was WRONG for the 4B MoE:
# an 8GB Mac passed it (8000 >= 4000) and spawned 2 workers — each holding
# its own model copy (~1.2-1.6GB) on a machine that can barely fit one.
# 16000MB keeps 2 workers on 16GB+ machines only (user report 2026-08-17:
# 2 workers on 8GB → OOM → 90s inference timeouts → Halim looked offline).
MIN_RAM_FOR_SECOND_WORKER = 16000  # MB — need 2.5GB per worker minimum

# ── Inference Priority Levels ─────────────────────────────────────────
# (duplicated above for documentation — these are the canonical definitions)

class _InferenceWorker:
    """Dedicated single-thread for MLX inference with priority queuing.

    MLX Metal stream contexts are NOT inherited by child threads.
    ThreadingHTTPServer runs each request on a new thread, so MLX inference
    must be serialized onto a single worker thread where Metal is
    initialized and the model is cached.

    CONCURRENCY (2026-08-14 upgrade):
      1. Priority queue: CRITICAL entry-qualification requests jump ahead of
         LOW-priority research/chat requests, so a 60s Halim research
         verdict never blocks a 30s entry qualification.
      2. Multi-process worker pool: when system RAM ≥ MIN_RAM_FOR_SECOND_WORKER
         MB, a second MLX worker PROCESS is spawned (MLX Metal can't share a
         stream across processes, but each process has its own). Requests
         are distributed round-robin across workers, doubling throughput.
         This is opt-in via HALIM_SERVE_WORKERS=2+ env var.
    """

    def __init__(self, n_workers: int = 1):
        self._n_workers = max(1, n_workers)
        self._queue = PriorityQueue()  # PriorityItem entries
        self._threads: List[threading.Thread] = []
        self._started = False
        self._model_loaded = False
        self._last_completed: float = time.time()
        self._stall_watch_started = False
        self._counter = 0  # monotonically increasing for FIFO within priority
        self._counter_lock = threading.Lock()
        # ── Metrics ─────────────────────────────────────────────────────
        self._req_total = 0
        self._req_served = 0
        self._metrics_lock = threading.Lock()
        # Track active workers for multi-process
        self._worker_procs: List[Process] = []

    def _next_seq(self) -> int:
        with self._counter_lock:
            self._counter += 1
            return self._counter

    def _start_stall_watchdog(self) -> None:
        """Daemon watchdog: force-complete a request stuck too long."""
        if self._stall_watch_started:
            return
        self._stall_watch_started = True
        # STALL FIX (2026-08-05): 120s was too tight — under memory pressure
        # legitimate 512-token generations exceed 120s, so the watchdog
        # force-completed requests that were actually being processed. 300s
        # only fires on a genuinely stuck worker.
        _stall_sec = float(os.environ.get("HALIM_SERVE_STALL_SEC", "300.0"))

        def _watch():
            while True:
                time.sleep(10)
                try:
                    if time.time() - self._last_completed < _stall_sec:
                        continue
                    # Worker silent too long — something may be stuck.
                    # Force-complete any pending future so callers unblock.
                    # Drain ALL pending items from the priority queue.
                    while True:
                        try:
                            _item = self._queue.get_nowait()
                            _prio, _seq, prompt, purpose, future, thinking = _item
                            future["result"] = {
                                "ok": False, "reason": "stall_recovered",
                                "message": f"Worker idle >{_stall_sec:.0f}s — request recovered",
                            }
                            future["done"] = True
                        except Empty:
                            break
                        except Exception:
                            pass
                        import gc as _gc
                        _gc.collect()
                        try:
                            import mlx.core as _mx
                            _mx.clear_cache()
                        except Exception:
                            pass
                        self._last_completed = time.time()
                except Exception:
                    pass

        threading.Thread(target=_watch, daemon=True,
                         name="halim-stall-watchdog").start()

    def model_ready(self) -> bool:
        return self._model_loaded

    def start(self):
        if not self._started:
            self._started = True
            # Start one thread per worker. On a single-worker system (8GB
            # Mac) all threads share the same process but the priority queue
            # ensures CRITICAL requests are served first. With HALIM_SERVE_WORKERS=2+,
            # each worker gets its own thread (MLX doesn't support true parallel
            # threads in-process, but the priority mechanism still works).
            for i in range(self._n_workers):
                t = threading.Thread(
                    target=self._run, daemon=True, name=f"halim-worker-{i}"
                )
                self._threads.append(t)
                t.start()
            self._start_stall_watchdog()

    def _run(self):
        """Worker loop: pull requests from priority queue, run inference."""
        # Initialize Metal on THIS thread (the only thread that will run MLX)
        try:
            import mlx.core as mx
            mx.set_default_device(mx.gpu)
            mx.clear_cache()
            import os as _os
            _cache_mb = int(_os.environ.get("MLX_GPU_CACHE_LIMIT_MB", "0"))
            if _cache_mb > 0:
                mx.set_cache_limit(_cache_mb * 1024 * 1024)
            mx.eval(mx.array([1.0]))  # Validate Metal stream
        except Exception:
            pass

        # Warmup: pre-load model on this thread.
        try:
            _warmup = complete_reasoning("Respond with the single word OK.", purpose="regime")
            if _warmup.get("ok"):
                self._model_loaded = True
        except Exception:
            pass

        while True:
            try:
                # Priority queue: (priority, seq, prompt, purpose, future, thinking)
                # Lower priority number = higher urgency
                _prio, _seq, prompt, purpose, future, thinking = self._queue.get()
                # DEAD-REQUEST SKIP (2026-08-17): the caller's 90s wait may
                # have already expired while this item sat in a deep queue.
                # Running it now is pure waste — the future is done (the
                # caller got inference_timeout and fell back). Drop it.
                if future.get("done"):
                    continue
                # Retry loading if warmup failed
                if not self._model_loaded:
                    _retry = complete_reasoning("Respond with the single word OK.", purpose="regime")
                    if _retry.get("ok"):
                        self._model_loaded = True
                result = complete_reasoning(prompt, purpose=purpose, thinking=thinking)
                future["result"] = result
                future["done"] = True
                self._last_completed = time.time()
                self._track_served()
            except Exception as exc:
                try:
                    future["result"] = {"ok": False, "reason": str(exc)[:200], "message": "Inference error"}
                    future["done"] = True
                    self._last_completed = time.time()
                except Exception:
                    pass

    def get_queue_size(self) -> int:
        """Current number of pending requests in the priority queue."""
        return self._queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        """Return inference worker stats for monitoring."""
        with self._metrics_lock:
            return {
                "n_workers": self._n_workers,
                "queue_size": self._queue.qsize(),
                "requests_total": self._req_total,
                "requests_served": self._req_served,
                "model_loaded": self._model_loaded,
            }

    def _track_request(self, priority: int):
        """Thread-safe request counter increment."""
        with self._metrics_lock:
            self._req_total += 1

    def _track_served(self):
        """Thread-safe served counter increment."""
        with self._metrics_lock:
            self._req_served += 1

    def complete(self, prompt: str, *, purpose: str = "reasoning", thinking: bool = False,
                 priority: int = PRIORITY_MEDIUM) -> Dict[str, Any]:
        """Run inference via priority queue. Non-blocking for the HTTP handler thread.

        Priority levels (lower = more urgent):
          PRIORITY_CRITICAL — entry qualification during active trading
          PRIORITY_HIGH     — regime classification, winrate diagnosis
          PRIORITY_MEDIUM   — thinking supervision, exit consultation
          PRIORITY_LOW      — research daemon, chat, code generation
        """
        import time as _t
        import os as _os

        # ── QUEUE CAP (2026-08-17, user report "Halim offline everywhere") ──
        # The priority queue was UNBOUNDED: when the 4B MoE on an 8GB Mac
        # couldn't generate faster than the bot submitted, the backlog grew
        # to 375+ pending requests. Every new caller then waited behind the
        # whole backlog and timed out at 90s — and every blocked HTTP
        # handler thread held a stack, blowing out wired memory → swap
        # thrash → generation slowed → queue grew MORE (death spiral).
        # Fix: reject fast at capacity — BUT only MEDIUM/LOW priority are
        # capped. CRITICAL (entry qualification) and HIGH (regime
        # classification) ALWAYS get through: those are the calls the bot
        # genuinely needs, and they are rare. The bot's async entry-debate
        # flood (16 req/min vs 3 served/min) otherwise fills the cap and
        # starves the regime classifier — regime stayed stuck on the
        # mechanical fallback (user report 2026-08-17: "0 Halim / 1
        # statistical classifications").
        _MAX_QUEUE = int(_os.environ.get("HALIM_SERVE_MAX_QUEUE", "6"))
        if priority <= PRIORITY_HIGH:
            pass  # critical/high always queue
        elif self._queue.qsize() >= _MAX_QUEUE:
            self._track_request(priority)
            return {
                "ok": False, "reason": "busy",
                "message": f"Halim at capacity ({_MAX_QUEUE} pending) — "
                            "advisory request skipped (caller falls back)",
            }

        # ── Memory gate ──────────────────────────────────────────────────
        # The old 900MB cap was tuned for Qwen3-1.7B. The 4B MoE legitimately
        # sits at ~1.2-1.6GB with a warm KV cache — 900MB blocked EVERY request
        # (memory_pressure 503). 2500MB is the swap-thrash ceiling on 8GB.
        try:
            import psutil as _ps
            _max_rss = int(_os.environ.get("HALIM_SERVE_MAX_RSS_MB", "2500"))
            _rss_mb = int(_ps.Process().memory_info().rss / (1024 * 1024))
            if _rss_mb > _max_rss:
                import gc as _gc
                _gc.collect()
                try:
                    import mlx.core as _mx
                    _mx.clear_cache()
                except Exception:
                    pass
                _rss2 = int(_ps.Process().memory_info().rss / (1024 * 1024))
                if _rss2 > _max_rss:
                    return {
                        "ok": False, "reason": "memory_pressure",
                        "message": f"Halim RSS {_rss2}MB > {_max_rss}MB cap — inference blocked",
                    }
        except Exception:
            pass

        self._track_request(priority)
        future: Dict[str, Any] = {"done": False, "result": None}
        self._queue.put((priority, self._next_seq(), prompt, purpose, future, thinking))
        # Wait with timeout. Qwen3.5-4B MoE (4-bit, mmap'd weights) can take
        # 30-90s for the FIRST real inference while pages fault in under
        # memory pressure — the hardcoded 30s caused "inference_timeout" even
        # though the model was loaded. Use HALIM_INFERENCE_TIMEOUT_SEC (env,
        # default 90) so cold-start inference isn't killed.
        _INFER_WAIT = float(_os.environ.get("HALIM_INFERENCE_TIMEOUT_SEC", "90.0"))
        deadline = _t.time() + _INFER_WAIT
        while not future.get("done") and _t.time() < deadline:
            _t.sleep(0.05)
        if future.get("done") and future.get("result"):
            return future["result"]
        return {"ok": False, "reason": "inference_timeout",
                "message": f"Inference timed out after {_INFER_WAIT:.0f}s"}


# Global inference worker (initialized in main())
_inference_worker: _InferenceWorker = None  # type: ignore[assignment]


def get_inference_stats() -> Dict[str, Any]:
    """Return inference worker stats (for telemetry)."""
    if _inference_worker is None:
        return {"n_workers": 0, "queue_size": 0, "model_loaded": False}
    return _inference_worker.get_stats()


def _detect_worker_count() -> int:
    """Determine the optimal number of worker threads based on system resources.

    On 8GB Mac: 1 worker (MLX can't parallelize Metal on a single GPU anyway,
                but priority queuing ensures fairness).
    On 16GB+ Mac: 2 workers (double throughput when RAM allows).
    Override via HALIM_SERVE_WORKERS env var.
    """
    _env = os.environ.get("HALIM_SERVE_WORKERS")
    if _env:
        try:
            return max(1, int(_env))
        except ValueError:
            pass
    try:
        import psutil as _ps
        _ram_mb = _ps.virtual_memory().total / (1024 * 1024)
    except Exception:
        _ram_mb = 8000
    if _ram_mb >= MIN_RAM_FOR_SECOND_WORKER:
        return min(2, max(1, cpu_count()))
    return 1


def _get_worker() -> _InferenceWorker:
    """Lazy-access the global worker."""
    global _inference_worker
    if _inference_worker is None:
        _inference_worker = _InferenceWorker(n_workers=_detect_worker_count())
    return _inference_worker


def _cleanup_serve_memory() -> float:
    """Free memory from Halim serve caches. Returns estimated MB freed."""
    freed = 0.0
    try:
        import gc
        gc.collect()
        freed += 3.0
    except Exception:
        pass
    try:
        import mlx.core as mx
        mx.clear_cache()
        freed += 10.0  # MLX GPU cache can be large
    except Exception:
        pass
    return freed


def _with_runtime(body: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(body)
    out["runtime"] = runtime_envelope()
    return out


def _record_action(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        root = os.getenv("HALIM_REPO_ROOT", "")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from core.halim_action_learn import record_action
        record_action(
            str(body.get("capability", "reasoning")),
            str(body.get("action", body.get("purpose", "record"))),
            input_text=str(body.get("input", body.get("prompt", "")))[:8000],
            output_text=str(body.get("output", body.get("text", "")))[:8000],
            outcome=str(body.get("outcome", "ok")),
            source=str(body.get("source", "halim_server")),
            meta=body.get("meta") if isinstance(body.get("meta"), dict) else None,
        )
        return {"ok": True, "recorded": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _export_gold() -> Dict[str, Any]:
    try:
        root = os.getenv("HALIM_REPO_ROOT", "")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from core.halim_action_learn import export_action_gold
        return {"ok": True, **export_action_gold()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _trigger_evolve() -> Dict[str, Any]:
    try:
        root = os.getenv("HALIM_REPO_ROOT", "")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from core.graceful_shutdown import flush_halim_data, flush_owned_brain
        from core.config import BotConfig
        cfg = BotConfig()
        halim = flush_halim_data(cfg, trigger="server_evolve")
        brain = flush_owned_brain(cfg, trigger="server_evolve", push_git=False)
        return {"ok": True, "halim": halim, "evolution": brain}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _unlock_ladder() -> Dict[str, Any]:
    try:
        root = os.getenv("HALIM_REPO_ROOT", "")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from core.halim_unlock import unlock_ladder
        return unlock_ladder()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _halim_chat(message: str, *, context: str = "", purpose: str = "chat") -> Dict[str, Any]:
    try:
        root = os.getenv("HALIM_REPO_ROOT", "")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from core.halim_chat import halim_chat
        return halim_chat(message, context=context, purpose=purpose)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _halim_generate(kind: str, prompt: str, *, path_hint: str = "") -> Dict[str, Any]:
    try:
        root = os.getenv("HALIM_REPO_ROOT", "")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        from core.halim_chat import halim_generate
        return halim_generate(kind, prompt, path_hint=path_hint)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


class HalimHandler(BaseHTTPRequestHandler):
    server_version = "HalimServe/1-active"

    def log_message(self, fmt: str, *args) -> None:
        if os.getenv("HALIM_SERVE_QUIET", "true").lower() not in ("1", "true", "yes"):
            super().log_message(fmt, *args)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _json(self, code: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(_with_runtime(body), default=str).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Halim-Runtime", "active")
            self.send_header("X-Halim-Inference-Only", "false")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            # Lightweight GPU cache flush after POST responses (throttled ~15s)
            # Prevents KV cache buildup during continuous RTH prompts.
            now_cache = time.time()
            last = getattr(self, "_last_cache_flush", 0.0)
            if now_cache - last >= 15.0:
                self._last_cache_flush = now_cache
                try:
                    import mlx.core as _mx
                    _mx.clear_cache()
                except Exception:
                    pass

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "model": MODEL_NAME, "protocol": PROTOCOL_VERSION})
        elif self.path == "/v1/status":
            st = collect_status()
            st["inference"] = _get_worker().get_stats()
            self._json(200, st)
        elif self.path == "/v1/runtime":
            self._json(200, {"ok": True, "model": MODEL_NAME, **runtime_envelope()})
        elif self.path == "/v1/unlock":
            self._json(200, _unlock_ladder())
        elif self.path == "/v1/manifest":
            st = collect_status()
            st["inference"] = _get_worker().get_stats()
            self._json(200, {"manifest": st, "protocol": PROTOCOL_VERSION})
        elif self.path == "/v1/stats":
            self._json(200, _get_worker().get_stats())
        else:
            self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        try:
            self._do_post()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as exc:
            try:
                self._json(500, {"ok": False, "error": str(exc)[:120]})
            except Exception:
                pass

    def _do_post(self) -> None:
        try:
            body = self._read_json()
        except Exception:
            self._json(400, {"ok": False, "error": "invalid_json"})
            return

        if self.path == "/v1/complete":
            prompt = str(body.get("prompt", ""))
            purpose = str(body.get("purpose", "reasoning"))
            # Trading-focus guard REMOVED — local Halim always available (policy: unlimited).
            _priority = str(body.get("priority", "")).lower()
            # entry_debate = ASYNC advisory debate (cache-TTL'd, non-blocking)
            # → MEDIUM. Only a TRUE blocking entry qualification
            # ("entry_qualification" / explicit critical) is PRIORITY_CRITICAL.
            # The bot debates every candidate ≥0.48 each cycle — marking all
            # of those CRITICAL bypassed the queue cap and starved regime
            # classification (user report 2026-08-17).
            if purpose == "entry_qualification" or "critical" in _priority:
                _prio = PRIORITY_CRITICAL
            elif "regime" in purpose or "diagnos" in purpose:
                _prio = PRIORITY_HIGH
            elif "exit" in purpose or "think" in purpose:
                _prio = PRIORITY_MEDIUM
            else:
                _prio = PRIORITY_LOW
            out = _get_worker().complete(prompt, purpose=purpose, priority=_prio)
            if out.get("ok") and out.get("text"):
                rec = _record_action({
                    "capability": purpose,
                    "action": "complete",
                    "input": prompt,
                    "output": out.get("text"),
                    "source": "halim_lm",
                })
                out["action_recorded"] = rec.get("recorded", False)
            code = 200 if out.get("ok") else 503
            self._json(code, out)
        elif self.path == "/v1/complete-thinking":
            prompt = str(body.get("prompt", body.get("input", "")))
            purpose = str(body.get("purpose", "reasoning"))
            out = _get_worker().complete(prompt, purpose=purpose, thinking=True,
                                          priority=PRIORITY_MEDIUM)
            if out.get("ok") and out.get("text"):
                rec = _record_action({
                    "capability": purpose,
                    "action": "complete_thinking",
                    "input": prompt,
                    "output": out.get("text"),
                    "source": "halim_lm_thinking",
                })
                out["action_recorded"] = rec.get("recorded", False)
            out["thinking"] = True
            code = 200 if out.get("ok") else 503
            self._json(code, out)
        elif self.path == "/v1/record":
            out = _record_action(body)
            self._json(200 if out.get("ok") else 500, out)
        elif self.path == "/v1/export":
            out = _export_gold()
            self._json(200 if out.get("ok") else 500, out)
        elif self.path == "/v1/evolve":
            out = _trigger_evolve()
            self._json(200 if out.get("ok") else 500, out)
        elif self.path == "/v1/chat":
            msg = str(body.get("message", body.get("prompt", "")))
            ctx = str(body.get("context", ""))
            purpose = str(body.get("purpose", "chat"))
            # Trading-focus guard REMOVED — local Halim always available (policy: unlimited).
            out = _halim_chat(msg, context=ctx, purpose=purpose)
            self._json(200 if out.get("ok") else 503, out)
        elif self.path == "/v1/generate":
            kind = str(body.get("kind", "code"))
            prompt = str(body.get("prompt", ""))
            out = _halim_generate(kind, prompt, path_hint=str(body.get("path", "")))
            self._json(200 if out.get("ok") else 503, out)
        else:
            self._json(404, {"ok": False, "error": "not_found"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} active runtime server")
    parser.add_argument("--host", default=os.getenv("HALIM_SERVE_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("HALIM_SERVE_PORT", DEFAULT_PORT)))
    args = parser.parse_args(argv)

    os.environ.setdefault("HALIM_REPO_ROOT", str(_repo_root()))
    ok, msg = enforce_active_runtime(context="halim serve")
    if not ok:
        print(f"⚠️  {msg}")

    # Pre-load the MLX model at startup so the first client request doesn't
    # trigger a slow lazy load inside a thread lock.
    print("📦 Pre-loading MLX model...", flush=True)
    # Initialize Metal device before model load (prevents stream errors)
    import os as _os
    if _os.getenv("HALIM_LM_BACKEND", "").lower() == "mlx":
        try:
            import mlx.core as _mx
            _mx.set_default_device(_mx.gpu)
            _mx.clear_cache()
            # Quick Metal validation — if this fails, MLX won't work
            _mx.eval(_mx.array([1.0]))
        except Exception:
            pass

    # Allow immediate port reuse (handles TIME_WAIT after rapid restart)
    import socketserver as _ss
    _ss.TCPServer.allow_reuse_address = True

    # Start the dedicated inference worker thread (MLX Metal can't cross threads)
    _n_workers = _detect_worker_count()
    print(f"   Starting {_n_workers} inference worker(s)...", flush=True)
    _get_worker().start()
    _st = _get_worker().get_stats()
    print(f"   Worker started (n_workers={_st['n_workers']})", flush=True)

    # Register with memory guardian (if available in path)
    try:
        if os.getenv("HALIM_REPO_ROOT", "") not in sys.path:
            sys.path.insert(0, os.getenv("HALIM_REPO_ROOT", ""))
        from core.memory_watchdog import register_component, ComponentPriority
        register_component(
            "halim_serve",
            priority=ComponentPriority.CRITICAL,  # NEVER kill Halim — it's the bot's brain
            cleanup_fn=lambda: _cleanup_serve_memory(),
            stuck_threshold_sec=120.0,
            metadata={"port": args.port, "model": MODEL_NAME},
        )
    except Exception:
        pass

    # Warmup: wait for worker thread to load model. Qwen3.5-4B MoE mmap'd
    # weights need ~10-15s cold on an 8GB Mac; under memory pressure (bot +
    # IB gateway resident) first fault-in can take 60s+. Default 90s matches
    # HALIM_INFERENCE_TIMEOUT_SEC — a 30s cap falsely reported "Model ready"
    # while the model was still page-faulting.
    import time as _wt
    _max_wait = float(_os.environ.get("HALIM_INFERENCE_TIMEOUT_SEC", "90.0"))
    print(f"   Warmup loop starting (max {_max_wait:.0f}s)...", flush=True)
    for _i in range(int(_max_wait * 2)):
        test = _get_worker().complete("hello", purpose="reasoning")
        # json_parse_failed means model loaded and generated — that's enough
        _ok = test.get("ok") or test.get("reason") == "json_parse_failed"
        if _ok:
            print(f"   ✅ Model ready ({test.get('backend', 'mlx')})", flush=True)
            break
        if "no_checkpoint" in str(test.get("reason", "")):
            print(f"   ⏭️  No checkpoint found — deferring load", flush=True)
            break
        _wt.sleep(0.5)
    else:
        print(f"   ⏳ Model not ready after {_max_wait:.0f}s — requests will lazy-load", flush=True)
    print(f"   About to start HTTP server...", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), HalimHandler)
    env = runtime_envelope()
    print(f"🧠 {MODEL_NAME} serve — http://{args.host}:{args.port}")
    print("   ACTIVE runtime — learns, records actions, writes owned weights (not Ollama read-only)")
    print("   Reflex (PPO/proxy) stays inline in HANOON")
    print("   GET  /health /v1/status /v1/runtime /v1/unlock")
    print("   POST /v1/complete /v1/record /v1/export /v1/evolve /v1/chat /v1/generate")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nHalim serve stopped.")
    return 0


def _repo_root() -> str:
    from pathlib import Path
    p = Path(__file__).resolve().parents[2]
    if (p / "models").is_dir():
        return str(p)
    return str(Path.cwd())


if __name__ == "__main__":
    sys.exit(main())
