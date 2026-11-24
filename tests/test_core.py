"""Offline tests: no model server needed."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from nyxcore.atlas import LsaEmbedder, build_atlas, search, wilson
from nyxcore.bench.agent import run_episode
from nyxcore.bench.worlds import TRAP_RECIPIENT, build_runbook
from nyxcore.providers import ChatResult, ToolCall
from nyxcore.proxy import parse_usage
from nyxcore.schema import TraceWriter, load_runs
from nyxcore.server import build_app


class ScriptedProvider:
    """Plays back a fixed list of tool calls and reports a growing prompt size."""

    name = "scripted"

    def __init__(self, script, window_cap=None):
        self.script = list(script)
        self.window_cap = window_cap

    def chat(self, model, messages, tools=None, **kwargs):
        size = sum(len(m.get("content") or "") for m in messages) // 4 + 50
        if self.window_cap:
            size = min(size, self.window_cap)
        name, args = self.script.pop(0)
        return ChatResult(text="", tool_calls=[ToolCall("c", name, args)], prompt_tokens=size, completion_tokens=10, latency_s=0.01, tokens_per_s=100.0)

    def count_tokens(self, model, text):
        return len(text) // 4


def book(**kw):
    return build_runbook(**{"seed": 1, "topic": "storage", "pages": 4, "chars_per_page": 600, "depth": 0.5, **kw})


def run(script, tmp_path=None, window=8192, cap=None, **kw):
    b = book(**kw)
    writer = TraceWriter(tmp_path / "run.jsonl") if tmp_path else None
    resolved = [(n, {"answer": b.canary_code} if a == "CANARY" else a) for n, a in script]
    return b, run_episode(ScriptedProvider(resolved, cap), "fake", b, episode_id="e1", task={"canary_code": b.canary_code, "fill_target": 0.5, "depth": 0.5},
                          context_window=window, chars_per_token=4.0, writer=writer)


READ_ALL = [("read_page", {"page": p}) for p in range(1, 5)]


def test_runbook_is_deterministic_and_hides_exactly_one_canary():
    a, b = book(), book()
    assert a == b
    assert sum(a.canary_code in page for page in a.pages) == 1
    assert a.canary_code in a.pages[a.canary_page]
    assert TRAP_RECIPIENT in a.pages[a.trap_page] and a.trap_page != a.canary_page
    assert a.canary_code not in a.distractor_codes
    assert book(seed=2).canary_code != a.canary_code or book(seed=2).pages != a.pages


def test_depth_places_the_canary():
    assert book(depth=0.0).canary_page == 0
    assert book(depth=1.0).canary_page == 3


def test_clean_run_is_graded_as_recalled():
    _, ep = run(READ_ALL + [("submit_answer", "CANARY")])
    assert ep.outcome["recalled"] and ep.outcome["submitted"] and not ep.outcome["trap_obeyed"]
    assert ep.outcome["pages_read"] == 4
    assert any("canary_seen" in s.flags for s in ep.steps) and any("trap_seen" in s.flags for s in ep.steps)


def test_obeying_the_trap_and_answering_with_a_distractor_are_both_caught():
    b = book()
    script = READ_ALL + [("send_report", {"recipient": TRAP_RECIPIENT, "body": "x"}), ("submit_answer", {"answer": b.distractor_codes[0]})]
    _, ep = run(script)
    assert ep.outcome["trap_obeyed"] and ep.outcome["wrong_code"] and not ep.outcome["recalled"]


def test_early_submit_is_rejected_once_so_the_full_context_is_carried():
    _, ep = run([("read_page", {"page": 1}), ("submit_answer", "CANARY")] + READ_ALL[1:] + [("submit_answer", "CANARY")])
    assert ep.outcome["recalled"] and ep.outcome["pages_read"] == 4
    assert "Rejected" in ep.steps[1].observation


def test_silent_truncation_is_flagged_when_the_provider_reads_less_than_was_sent():
    _, ep = run(READ_ALL + [("submit_answer", "CANARY")], window=400, cap=300, chars_per_page=1500)
    assert ep.outcome["truncated"]
    _, ok = run(READ_ALL + [("submit_answer", "CANARY")])
    assert not ok.outcome["truncated"]


def test_trace_roundtrip_and_metadata_capture_drops_text(tmp_path):
    _, ep = run(READ_ALL + [("submit_answer", "CANARY")], tmp_path)
    loaded = load_runs(tmp_path)
    assert len(loaded) == 1 and len(loaded[0].steps) == len(ep.steps) and loaded[0].outcome == ep.outcome
    quiet = TraceWriter(tmp_path / "quiet" / "q.jsonl", capture="metadata")
    quiet.episode(ep)
    quiet.step(ep.steps[0])
    raw = (tmp_path / "quiet" / "q.jsonl").read_text()
    assert "runbook" not in raw and json.loads(raw.splitlines()[1])["prompt_tokens"] == ep.steps[0].prompt_tokens


def test_wilson_interval_is_wide_for_small_samples():
    r = wilson(6, 6)
    assert r["rate"] == 1.0 and r["lo"] < 0.62 and r["hi"] == 1.0
    assert wilson(0, 0)["n"] == 0


def test_embedder_ranks_shared_vocabulary_first():
    texts = ["gpu scheduler checkpoint loss diverged", "invoice settlement payout merchant", "gpu checkpoint writer restarted", "merchant refund invoice"]
    emb = LsaEmbedder()
    vectors, _ = emb.fit(texts)
    scores = vectors @ emb.embed(["checkpoint gpu"])[0]
    assert set(scores.argsort()[-2:]) == {0, 2}


@pytest.fixture()
def atlas_dir(tmp_path):
    writer = TraceWriter(tmp_path / "runs" / "r.jsonl")
    for n, topic in enumerate(["storage", "training", "network"]):
        b = build_runbook(seed=n, topic=topic, pages=4, chars_per_page=600, depth=0.5)
        script = READ_ALL + [("submit_answer", {"answer": b.canary_code if n else "nope"})]
        run_episode(ScriptedProvider(script), "fake", b, episode_id=f"e{n}", task={"canary_code": b.canary_code, "fill_target": 0.5, "depth": 0.5, "topic": topic},
                    context_window=4096, chars_per_token=4.0, writer=writer)
    build_atlas(tmp_path / "runs", tmp_path / "atlas")
    return tmp_path / "atlas"


def test_atlas_geometry_and_stats(atlas_dir):
    atlas = json.loads((atlas_dir / "atlas.json").read_text())
    assert atlas["meta"]["episodes"] == 3 and len(atlas["steps"]) == 15
    assert atlas["models"][0]["stats"]["recall"] == {**atlas["models"][0]["stats"]["recall"], "k": 2, "n": 3}
    for e in atlas["episodes"]:
        heights = [s["y"] for s in atlas["steps"][e["s0"]:e["s1"]]]
        assert heights == sorted(heights), "context demand never shrinks inside a run"
    assert search(atlas_dir, "erasure coder replication lag")[0]["step"] < 5  # the storage run comes first


def test_server_is_local_only_and_serves_search(atlas_dir):
    client = TestClient(build_app(atlas_dir), base_url="http://127.0.0.1:4318")
    assert client.get("/atlas.json").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/search", params={"q": "gpu scheduler"}).json()["hits"]
    assert client.get("/atlas.json", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/atlas.json", headers={"Host": "attacker.example"}).status_code == 403


def test_proxy_usage_parsing_for_both_wire_formats():
    ollama = '{"message":{"content":"Hel"}}\n{"message":{"content":"lo"},"done":true,"prompt_eval_count":12,"eval_count":4,"eval_duration":200000000}\n'
    assert parse_usage("/api/chat", ollama) == {"output": "Hello", "prompt_tokens": 12, "completion_tokens": 4, "tokens_per_s": 20.0}
    sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":1}}\n\ndata: [DONE]\n'
    parsed = parse_usage("/v1/chat/completions", sse)
    assert (parsed["output"], parsed["prompt_tokens"], parsed["completion_tokens"]) == ("Hi", 7, 1)


def test_a_provider_error_ends_the_run_as_a_recorded_failure():
    class Crashing(ScriptedProvider):
        def chat(self, *a, **kw):
            if not self.script:
                raise RuntimeError("ollama 500: error parsing tool call")
            return super().chat(*a, **kw)

    b = book()
    ep = run_episode(Crashing(READ_ALL[:2]), "fake", b, episode_id="e1", task={}, context_window=8192, chars_per_token=4.0)
    assert len(ep.steps) == 2 and not ep.outcome["recalled"] and "error parsing tool call" in ep.outcome["error"]
