"""tests/test_halim_evidence — evidence-grounded HALIM chain-of-thought."""
from __future__ import annotations

from hanoon_prime.brain.halim_evidence import (
    build_evidence_prompt,
    collect_evidence,
    fetch_evidence_recs,
    maybe_fetch_evidence_recs,
    read_eval_tail,
)


def _line(token: str) -> str:
    return f"10:00:00.000  INFO   juli              EVAL {token}\n"


_LINES = [
    _line("AAPL:VETOED(0.700,SHORT)[trading_policy:direction_rejected]"),
    _line("TSLA:VETOED(0.650,SHORT)[trading_policy:direction_rejected]"),
    _line("AEON:VETOED(0.600,-)[trading_policy:low_penny_score]"),
    _line("NVD:BUY(0.850,L)[trading_policy:enter]"),
]


def _fake_query_factory(payload: str):
    def _q(base_url: str, prompt: str) -> dict:
        return {"text": payload}

    return _q


def test_collect_evidence_tally():
    ev = collect_evidence(_LINES)
    assert ev["eval_lines"] == 4
    assert ev["vetoes"] == 3
    assert ev["by_reason"] == {"direction_rejected": 2, "low_penny_score": 1}
    assert ev["top_veto_reason"] == "direction_rejected"
    assert ev["direction_rejected"] == 2
    assert ev["direction_rejected_mean_abs_score"] == 0.675
    assert ev["real_conviction_discarded"] is True


def test_collect_evidence_ignores_non_eval_lines():
    ev = collect_evidence(["not an eval line\n", "garbage\n"])
    assert ev["eval_lines"] == 0
    assert ev["vetoes"] == 0
    assert ev["top_veto_reason"] is None


def test_collect_evidence_realized_snapshot():
    snapshot = {"conf_wins": [3, 1], "conf_losses": [2, 8]}
    ev = collect_evidence([], realized_snapshot=snapshot, regime="trending_bullish")
    assert ev["regime"] == "trending_bullish"
    assert ev["win_rate"] == round(4 / 14, 3)
    assert ev["worst_conf_bin_losses"] == 8


def test_build_evidence_prompt_contains_signal():
    ev = collect_evidence(_LINES)
    prompt = build_evidence_prompt(ev)
    assert "direction_rejected" in prompt
    assert "0.675" in prompt  # mean abs conviction behind the rejections
    assert "top_veto_reason" in prompt


def test_read_eval_tail(tmp_path):
    log = tmp_path / "hanoon.log"
    log.write_text("".join(_LINES))
    tail = read_eval_tail(str(log), n=2)
    assert len(tail) == 2
    assert tail[-1].endswith("enter]\n")  # NVD buy is the last line


def test_read_eval_tail_missing_file():
    assert read_eval_tail("/no/such/file.log", n=10) == []


def test_fetch_evidence_recs_validates_bounds():
    ev = collect_evidence(_LINES)
    payload = (
        '{"reasoning":"ok","recommendations":['
        '{"action":"adjust_weight","param":"rsi","value":0.5,"reason":"x"},'
        '{"action":"adjust_threshold","param":"threshold","value":0.30,"reason":"bad"}'
        "]}"
    )
    recs = fetch_evidence_recs("http://x", ev, query=_fake_query_factory(payload))
    assert len(recs) == 1  # 0.30 threshold is below the 0.40 floor -> rejected
    assert recs[0]["param"] == "rsi"


def test_maybe_fetch_disabled_returns_empty():
    assert maybe_fetch_evidence_recs("http://x", {}, enabled=False) == []


def test_maybe_fetch_handles_failure(monkeypatch):
    from hanoon_prime.brain import halim_evidence as he

    def _boom(base_url: str, prompt: str) -> dict:
        raise RuntimeError("halim down")

    monkeypatch.setattr(he, "_http_query", _boom)
    assert (
        maybe_fetch_evidence_recs("http://x", collect_evidence(_LINES), enabled=True)
        == []
    )
